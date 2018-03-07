import numba as nb
import numpy as np
from scipy import optimize
import calibration
import os
import pickle
import lab
import cross as cs # conflict with cross()

def make_von_neumann(density, domain, max_cycles=100000):
    """
    density = positive function float -> float
    domain = [a, b] domain of density
    """
    out = optimize.minimize_scalar(lambda x: -density(x), bounds=domain, method='bounded')
    if not out.success:
        raise RuntimeError('cannot find the maximum of density in [%g, %g]' % domain)
    top = density(out.x)
    left = domain[0]
    right = domain[1]
    
    if not isinstance(density, nb.targets.registry.CPUDispatcher):
        density = nb.jit('f8(f8)', nopython=True)(density)
    
    @nb.jit('f8', nopython=True)
    def von_neumann():
        i = 0
        while i < max_cycles:
            candidate = np.random.uniform(left, right)
            height = np.random.uniform(0, top)
            if height <= density(candidate):
                return candidate
            i += 1
        return np.nan
    
    return von_neumann

@nb.jit('f8[3](f8[3],f8[3])', nopython=True, cache=True)
def cross(a, b):
    """
    cross product a x b
    """
    return np.array([
        a[1]*b[2] - a[2]*b[1],
        a[2]*b[0] - a[0]*b[2],
        a[0]*b[1] - a[1]*b[0]
    ])

@nb.jit('f8[3](f8[3])', nopython=True, cache=True)
def versor(a):
    """
    normalize a (returns a / |a|)
    """
    return a / np.sqrt(np.sum(a ** 2))

@nb.jit('f8(f8,f8,f8)', nopython=True, cache=True)
def compton_photon_energy(E, cos_theta, m_e):
    return E / (1 + E / m_e * (1 - cos_theta))

@nb.jit('f8(f8,f8,f8)', nopython=True, cache=True)
def klein_nishina(E, cos_theta, m_e):
    """
    E = photon energy [MeV]
    cos_theta = cosine of polar angle
    
    from https://en.wikipedia.org/wiki/Klein–Nishina_formula
    """
    P = compton_photon_energy(E, cos_theta, m_e) / E
    return 1/2 * P**2 * (P + P**-1 - (1 - cos_theta**2))

@nb.jit('f8[:](u4,i8)', nopython=True, cache=True)
def random_normal(n, seed=-1):
    out = np.empty(n)
    if seed >= 0:
        np.random.seed(seed)
    for i in range(len(out)):
        out[i] = np.random.normal()
    return out

def energy_nai(E, *a, **k):
    seed = k.pop('seed', -1)
    E_cal = calibration.energy_calibration(*a, **k)(E)
    sigma = calibration.energy_sigma(*a, **k)(E)
    return E_cal + random_normal(len(E), seed=seed) * sigma

@nb.jit(nopython=True, cache=True)
def mc(energy, theta_0=0, N=1000, seed=-1, beam_sigma=1.74, beam_center=-0.09, nai_distance=40, nai_radius=2.54, nai_depth=5.08, nai_density=3.667, m_e=0.511, acc_bounds=True, geometry_3d=True, max_secondary_cos_theta=1):
    """
    Simulate Compton scattering on the target and energy measurement of scattered photon with NaI.
    
    Parameters
    ----------
    energy : float
        Energy of beam photons, in MeV.
    theta_0 : float
        Position of NaI as polar angle from experiment axis, in degrees.
    N : integer
        Number of output montecarlo samples.
    seed : integer
        Seed for random numbers generation. If negative, the state of the generator is left unchanged.
    beam_sigma : float
        Width of the beam as gaussian standard deviation, in degrees.
    beam_center : float
        Center of the beam as polar angle from experiment axis, in degrees.
    nai_distance : float
        Distance of the NaI from the target, in cm.
    nai_radius : float
        Radius of the NaI, in cm.
    nai_depth : float
        Depth of the NaI, i.e. distance from side facing the target to the other side, in cm.
    nai_density : float
        Density of NaI, in g/cm^3.
    m_e : float
        Electron mass, in MeV.
    geometry_3d : bool
        If False, use a simplified 2d model of the NaI without keeping into account
        interaction probabilities of the photon.
    acc_bounds : bool
        If False, disable quick bounds on NaI shape that speed up montecarlo.
    
    Returns
    -------
    primary_photons : (N) float array
        Energies of Compton photons from target hitting the NaI.
    secondary_electrons : (N) float array
        Kinetical energies of electrons emitted from Compton in the NaI.
    weights_pp : (N) float array
    weights_se : (N) float array
        Weights of primary photons and secondary electron samples, respectively.
        See for example numpy.histogram for the meaning.
    """
    # convert from degrees to radians
    beam_sigma *= np.pi / 180
    beam_center *= np.pi / 180
    theta_0 *= np.pi / 180
    
    # axes of the source
    X = np.array([1., 0, 0])
    Y = np.array([0, 1., 0])
    Z = np.array([0, 0, 1.])
    
    # axes of the NaI
    z = Z * np.cos(theta_0) + X * np.sin(theta_0)
    y = Y
    x = X * np.cos(theta_0) - Z * np.sin(theta_0)
    
    # seed
    if seed >= 0:
        np.random.seed(seed)
    
    primary_photons = np.empty(N)
    secondary_electrons = np.empty(N)
    if geometry_3d:
        weights_pp = np.empty(N)
        weights_se = np.empty(N)
    else:
        weights_pp = np.ones(N)
        weights_se = np.ones(N)
    i = 0
    count = 0
    while i < N:
        # angular coordinates of the beam photon
        theta_f = np.random.normal(loc=beam_center, scale=beam_sigma)
        phi_f = np.random.uniform(0, 2 * np.pi)
        
        # axis of the beam photon
        # x_f and y_f computed together with acceptance bounds
        z_f = Z * np.cos(theta_f) + np.sin(theta_f) * (X * np.cos(phi_f) + Y * np.sin(phi_f))
        
        # default values for phi bounds
        phi_min = 0
        phi_max = 2 * np.pi
        
        # default values for theta bounds
        cos_theta_min = -1
        cos_theta_max = 1
        
        # default value for y axis of beam photon reference system
        # I assume that X and z_f are not parallel
        y_f = versor(cross(z_f, X))
        
        if acc_bounds:
            # compute acceptance bounds for theta of the compton photon
            cos_theta_rel = np.dot(z, z_f)
            sin_theta_rel = np.sqrt(1 - cos_theta_rel**2)
            cos_vartheta = nai_distance / np.sqrt(nai_distance**2 + nai_radius**2)
            sin_vartheta = nai_radius / np.sqrt(nai_distance**2 + nai_radius**2)
            cos_theta_plus = cos_theta_rel * cos_vartheta - sin_theta_rel * sin_vartheta
            cos_theta_minus = cos_theta_rel * cos_vartheta + sin_theta_rel * sin_vartheta
            if -cos_vartheta <= cos_theta_rel <= cos_vartheta:
                # the NaI do not overlap the z_f axis
                cos_theta_min = min(cos_theta_plus, cos_theta_minus)
                cos_theta_max = max(cos_theta_plus, cos_theta_minus)
                y_f = versor(cross(z_f, z)) # such that x_f points to the side of the NaI
                phi_max = np.arctan(nai_radius / (nai_distance * sin_theta_rel))
                phi_min = -phi_max
            elif cos_theta_rel > cos_vartheta:
                # NaI in front of beam
                cos_theta_min = min(cos_theta_plus, cos_theta_minus)
                cos_theta_max = 1
            else:
                # NaI behind beam
                cos_theta_min = -1
                cos_theta_max = max(cos_theta_plus, cos_theta_minus)
        
        # x axis of beam photon reference system
        x_f = versor(cross(y_f, z_f))
        
        # extract theta of compton photon with von neumann
        while 1:
            cos_theta_candidate = np.random.uniform(cos_theta_min, cos_theta_max)
            # to compute kn_max I assume that klein_nishina has only one minimum
            kn_max = max(klein_nishina(energy, cos_theta_min, m_e), klein_nishina(energy, cos_theta_max, m_e))
            von_neumann = np.random.uniform(0, kn_max)
            if von_neumann <= klein_nishina(energy, cos_theta_candidate, m_e):
                break
        cos_theta = cos_theta_candidate
        sin_theta = np.sqrt(1 - cos_theta**2)
        phi = np.random.uniform(phi_min, phi_max)
        
        rho = z_f * cos_theta + sin_theta * (x_f * np.cos(phi) + y_f * np.sin(phi))
        rho_z = np.dot(rho, z)
        radius = nai_distance * np.sqrt(1 - rho_z**2) / rho_z
        
        # condition: the photon intersects the front face of the NaI
        if 0 <= radius <= nai_radius:
            primary_photon = compton_photon_energy(energy, cos_theta, m_e)
            
            if geometry_3d:
                # compute the length of the trajectory of the photon inside the NaI
                # (assuming it wouldn't interact)
                radius_back = (nai_distance + nai_depth) / nai_distance * radius
                l = np.sqrt(nai_depth**2 + (radius_back - radius)**2)
                if radius_back > nai_radius:
                    l *= (nai_radius - radius) / (radius_back - radius)
            
                # compute probabilities of interaction
                weights_pp[i] = 1 - np.exp(-l * cs.photoel(primary_photon) * nai_density)
                weights_se[i] = 1 - np.exp(-l * cs.compton(primary_photon) * nai_density)
            
            # extract theta of secondary compton photon
            while 1:
                cos_theta_candidate = np.random.uniform(-1, max_secondary_cos_theta)
                kn_max = max(klein_nishina(primary_photon, -1, m_e), klein_nishina(primary_photon, max_secondary_cos_theta, m_e))
                von_neumann = np.random.uniform(0, kn_max)
                if von_neumann <= klein_nishina(primary_photon, cos_theta_candidate, m_e):
                    break
            cos_theta = cos_theta_candidate
            
            secondary_photon = compton_photon_energy(primary_photon, cos_theta, m_e)
            secondary_electron = primary_photon - secondary_photon
            
            primary_photons[i] = primary_photon
            secondary_electrons[i] = secondary_electron

            i += 1
        count += 1
        
    return primary_photons, secondary_electrons, weights_pp, weights_se

def mc_cal(*args, **kwargs):
    """
    Same as mc, but applies calibration.
    """
    p, s, wp, ws = mc(*args, **kwargs)
    p = energy_nai(p)
    s = energy_nai(s)
    return p, s, wp, ws

def mc_cached(*args, **kwargs):
    """
    Calls mc and saves results so that if called again
    with the same parameters it will not run again the Monte Carlo.
    
    Keyword arguments
    -----------------
    The argument <seed> must be specified and be >= 0,
    otherwise the result depends on the internal state
    of the sampler and can not be uniquely determined
    from the arguments.
    
    Additionally to those of mc, it recognises:
    
    calibration : bool (default: True)
        Apply calibration.
    """
    
    if not ('seed' in kwargs):
        raise ValueError('Argument <seed> not specified.')
    if kwargs['seed'] < 0:
        raise ValueError('Argument seed={} must be >= 0.'.format(kwargs['seed']))
    
    # load or create database
    directory = 'mc9_cache'
    database_file = '{}/mc9_cache.pickle'.format(directory)
    if not os.path.exists(directory):
        os.mkdir(directory)
    if os.path.exists(database_file):
        with open(database_file, 'rb') as file:
            database = pickle.load(file)
    else:
        database = {}
        
    calibration = kwargs.pop('calibration', True)
    
    # get result from cache or compute and save
    hashable_args = (args, frozenset(kwargs.items()))
    if hashable_args in database:
        data_file = database[hashable_args]
        data = np.load(data_file)
        p = data['primary']
        s = data['secondary']
        wp = data['weights_p']
        ws = data['weights_s']
    else:
        p, s, wp, ws = mc(*args, **kwargs)
        new_file = lab.nextfilename('mc9_cache', '.npz', prepath=directory)
        np.savez(new_file, primary=p, secondary=s, weights_p=wp, weights_s=ws)
        database[hashable_args] = new_file
    
    # save database
    with open(database_file, 'wb') as file:
        pickle.dump(database, file)
    
    # apply calibration
    if calibration:
        p = energy_nai(p, seed=4000000000 + kwargs['seed'])
        s = energy_nai(s, seed=2000000000 + kwargs['seed'])
    
    return p, s, wp, ws

if __name__ == '__main__':
    N=1000000
    p, s, wp, ws = mc_cached(1.33, theta_0=60, N=N, seed=0)
    
    from matplotlib.pyplot import *
    figure('mc9')
    clf()
    hist(p, bins=int(np.sqrt(N)), weights=wp / (N * (np.max(p) - np.min(p))), histtype='step', label='photoel')
    hist(s, bins=int(np.sqrt(N)), weights=ws / (N * (np.max(s) - np.min(s))), histtype='step', label='compton')
    legend(loc=1)
    show()
