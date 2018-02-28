import numba as nb
import numpy as np
from scipy import optimize

m_e = 0.511 # electron mass [MeV]

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
        density = nb.jit(density, nopython=True)
    
    @nb.jit(nb.float64(), nopython=True)
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

@nb.jit(nb.float64[3](nb.float64[3], nb.float64[3]), nopython=True, cache=True)
def cross(a, b):
    """
    cross product a x b
    """
    return np.array([
        a[1]*b[2] - a[2]*b[1],
        a[2]*b[0] - a[0]*b[2],
        a[0]*b[1] - a[1]*b[0]
    ])

@nb.jit(nb.float64[3](nb.float64[3]), nopython=True, cache=True)
def versor(a):
    """
    normalize a (returns a / |a|)
    """
    return a / np.sqrt(np.sum(a ** 2))

@nb.jit(nb.float64(nb.float64, nb.float64), nopython=True, cache=True)
def klein_nishina(E, cos_theta):
    """
    E = photon energy [MeV]
    cos_theta = cosine of polar angle
    
    from https://en.wikipedia.org/wiki/Klein–Nishina_formula
    """
    P = 1 / (1 + E / m_e * (1 - cos_theta))
    return 1/2 * P**2 * (P + P**-1 - (1 - cos_theta**2))

@nb.jit(nb.float64(nb.float64), nopython=True)
def energy_sigma(E):
    """
    E = energy [MeV]
    energy resolution of NaI + PMT
    TO BE MODIFIED
    also: find the reference
    """
    return (2.27 + 7.28 * E ** -0.29 - 2.41 * E ** 0.21) * E / (100 * 2.35)

@nb.jit(nb.float64(nb.float64), nopython=True)
def energy_calibration(E):
    """
    E = energy [MeV]
    TO BE MODIFIED
    returns non-calibrated energy
    """
    return E

@nb.jit(nopython=True, cache=True)
def mc(energy, theta_0=0, N=1000, seed=-1, beam_sigma=0, beam_center=0, nai_distance=20, nai_radius=2, energy_cal=True, energy_res=True, acc_bounds=True):
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
    
    if seed >= 0:
        np.random.seed(seed)
    out_energy = np.empty(N)
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
            kn_max = max(klein_nishina(energy, cos_theta_min), klein_nishina(energy, cos_theta_max))
            von_neumann = np.random.uniform(0, kn_max)
            if von_neumann <= klein_nishina(energy, cos_theta_candidate):
                break
        cos_theta = cos_theta_candidate
        sin_theta = np.sqrt(1 - cos_theta**2)
        phi = np.random.uniform(phi_min, phi_max)
        
        rho = z_f * cos_theta + sin_theta * (x_f * np.cos(phi) + y_f * np.sin(phi))
        rho_z = np.dot(rho, z)
        radius2 = nai_distance**2 * (1 - rho_z**2) / rho_z**2
        
        if radius2 <= nai_radius**2 and rho_z >= 0:
            out_energy[i] = energy / (1 + energy / m_e * (1 - cos_theta))
            if energy_res:
                out_energy[i] += np.random.normal(loc=0, scale=energy_sigma(out_energy[i]))
            if energy_cal:
                out_energy[i] = energy_calibration(out_energy[i])
            i += 1
        count += 1
    
    return out_energy, count

if __name__ == '__main__':
    from matplotlib.pyplot import *
    N = 10000
    energy, count = mc(1.33, theta_0=0, N=N, beam_sigma=0, seed=1, acc_bounds=True, energy_res=False)
    energy2, count2 = mc(1.33, theta_0=0, N=N, beam_sigma=0, acc_bounds=False, energy_res=False)

    figure('mc9')
    clf()
    hist(np.concatenate([energy]), bins='sqrt', histtype='step', label='N=%d' % (N,), density=True)
    hist(energy2, bins='sqrt', histtype='step', label='no acc bounds', density=True)
    # theta = np.linspace(0, np.pi, 1000)
    # kn = klein_nishina(0.662, theta)
    # plot(np.cos(theta), kn, '-k', label='E=.662')
    legend(loc=1)
    show()
