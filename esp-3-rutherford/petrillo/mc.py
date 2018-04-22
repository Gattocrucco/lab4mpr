import numba
import matplotlib.pyplot as plt
import numpy as np
from scipy import optimize
import os
import pickle
import lab
import dedx

# alpha particle properties
z = 2
A = 4

@numba.jit(nopython=True, cache=True)
def multiple_scattering(T, x):
    """
    non-relativistic
    scales as charge of particle
    
    Parameters
    ----------
    T : [MeV]
    x : [radiation lenghts]
    
    Returns
    -------
    theta_0_rms : [radians]
    """
    theta_0 = 6.8 / T * z * np.sqrt(x) * (1 + 0.038 * np.log(x))
    return max(theta_0, 0) # 0 last: NaN-propagating

@numba.jit(nopython=True, cache=True)
def energy_loss(T, x, rho, d_step=0.5, spread=0.1):
    """
    T : [MeV]
    x : [um]
    rho : [g/cm^3]
    d_step : [um]
    spread : in [0, 0.25]
    
    Returns
    -------
    New energy
    """
    d = 0
    T_in = T
    while d < x:
        if T < dedx.dedx_min:
            return 0
        T -= dedx.dedx(T) * rho / 10000 * d_step # 10000 is conversion MeV/cm -> MeV/um
        d += d_step
    if T < dedx.dedx_min:
        return 0
    T -= dedx.dedx(T) * rho / 10000 * (x - (d - d_step))
    T = np.random.normal(loc=T, scale=spread * (T_in - T))
    return max(T, 0)

@numba.jit(nopython=True, cache=True)
def theta_min(T, Z):
    """
    T : [MeV]
    """
    alpha = 1/137
    hc = 197.3 # MeV fm
    rbohr = 0.529e5 # fm
    return z * Z * alpha * hc / (T * rbohr)

@numba.jit(nopython=True, cache=True)
def T_out_factor(nucl_A, cos_theta):
    return 1 - 2 * A / nucl_A * (1 - cos_theta)

@numba.jit(nopython=True, cache=True)
def random_sign():
    return np.random.randint(2) * 2 - 1

# target material properties
X0_au = 0.33 # cm
X0_al = 8.9 # cm

rho_au = 19.320 # g/cm^3
rho_al = 2.699 # g/cm^3

Z_au = 79
Z_al = 13

A_au = 200
A_al = 27

# keyword arguments for available targets
target_au5 = dict(target_rho=rho_au, target_X0=X0_au, target_Z=Z_au, target_A=A_au, target_thickness=5.0)

@numba.jit(nopython=True, cache=True)
def mc(seed=-1, N=1000, amax=2.5, L=28.5, D=31.0, target_thickness=5.0, T=5.46, source_thickness=3.0, theta_eps=1, target_rho=rho_au, target_X0=X0_au, target_Z=Z_au, target_A=A_au):
    """
    Parameters
    ----------
    seed : integer
        Seed for random numbers generation. If negative, the state of the generator is left unchanged.
    N : integer >= 1
        Number of samples drawn.
    amax : float [mm]
        Half width of the collimator.
    L : float [mm]
        Distance center-detector.
    D : float [mm]
        Distance center-source.
    target_thickness : float [um]
        Thickness of target.
    T : float [MeV]
        Initial kinetical energy of alpha particles.
    source_thickness : float [um]
        Thickness of encapsulation of source.
    theta_eps : float [degrees]
        Minimum angle considered for Rutherford scattering.
    target_rho : float [g/cm^3]
        Density of target.
    target_X0 : float [cm]
        Radiation length of target.
    
    Returns
    -------
    theta
    weight
    energy
    """
    # seed
    if seed >= 0:
        np.random.seed(seed)
    
    # return arrays
    thetas = np.empty(N)
    weights = np.empty(N)
    energies = np.empty(N)
    
    # rutherford scattering setup
    theta_eps = np.radians(theta_eps)
    y_max = 1 / (1 - np.cos(theta_eps)) if theta_eps > 0.01 else 2 / theta_eps**2
    
    i = 0
    while i < N:
        # energy loss in the encapsulation
        T_beam = energy_loss(T, source_thickness, rho_au)
        if T_beam == 0:
            continue
        
        # extract beam from source
        theta_source_max = np.arctan(amax / D)
        theta_source = np.random.uniform(-theta_source_max, theta_source_max)
        
        # hit point on the target
        a = D * np.tan(theta_source)
        
        # depth of rutherford scattering point in the target
        # the scattering is actually more probable toward the end of the trajectory because of energy loss
        rutherford_depth = np.random.uniform(0, target_thickness)
        
        # multiple scattering and energy loss before rutherford scattering
        depth_before = rutherford_depth / np.cos(theta_source)
        ms_before = np.random.normal(loc=0, scale=multiple_scattering(T_beam, depth_before / 10000 / target_X0))
        T_rutherford = energy_loss(T_beam, depth_before, target_rho)
        if T_rutherford == 0:
            continue
        
        # rutherford scattering
        y = np.random.uniform(1, y_max)
        cos_theta_rutherford = 1 - 1/y
        theta_rutherford = np.arccos(cos_theta_rutherford) * random_sign()
        T_scattered = T_rutherford * T_out_factor(target_A, cos_theta_rutherford)
        if T_scattered == 0:
            continue
        
        # multiple scattering and energy loss after rutherford scattering
        if abs(theta_source + ms_before + theta_rutherford) > np.radians(85):
            continue # would require different treatment
        depth_after = (target_thickness - rutherford_depth) / np.cos(theta_source + ms_before + theta_rutherford)
        ms_after = np.random.normal(loc=0, scale=multiple_scattering(T_scattered, depth_after / 10000 / target_X0))
        T_out = energy_loss(T_scattered, depth_after, target_rho)
        if T_out == 0:
            continue
        
        # compute intersection with circumference of detector
        theta_final = theta_source + ms_before + theta_rutherford + ms_after
        if abs(theta_final) > np.radians(85):
            continue
        tan_theta_final = np.tan(theta_final)
        sin_theta = (a/L + tan_theta_final * np.sqrt(1 + tan_theta_final**2 - (a/L)**2)) / (1 + tan_theta_final**2)
        theta = np.arcsin(sin_theta)
        
        thetas[i] = theta
        weights[i] = 1 / T_rutherford ** 2
        energies[i] = T_out
        
        i += 1
               
    return thetas, weights, energies

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
    """
    
    if not ('seed' in kwargs):
        raise ValueError('Argument <seed> not specified.')
    if kwargs['seed'] < 0:
        raise ValueError('Argument seed={} must be >= 0.'.format(kwargs['seed']))
    
    # load or create database
    directory = 'mc_cache'
    database_file = '{}/mc_cache.pickle'.format(directory)
    if not os.path.exists(directory):
        os.mkdir(directory)
    if os.path.exists(database_file):
        with open(database_file, 'rb') as file:
            database = pickle.load(file)
    else:
        database = {}
        
    # get result from cache or compute and save
    hashable_args = (args, frozenset(kwargs.items()))
    if hashable_args in database:
        data_file = database[hashable_args]
        data = np.load(data_file)
        t = data['t']
        w = data['w']
        e = data['e']
    else:
        t, w, e = mc(*args, **kwargs)
        new_file = lab.nextfilename('mc_cache', '.npz', prepath=directory)
        np.savez(new_file, t=t, w=w, e=e)
        database[hashable_args] = new_file
    
    # save database
    with open(database_file, 'wb') as file:
        pickle.dump(database, file)
    
    return t, w, e

if __name__ == '__main__':
    t, w, e = mc_cached(seed=0, N=10000000, theta_eps=5)
    
    fig = plt.figure('mc')
    fig.clf()
    fig.set_tight_layout(True)
    
    ax = fig.add_subplot(111)
    ax.hist(np.degrees(t), bins=int(np.sqrt(len(t))), weights=w, histtype='step', density=True)
    ax.set_xlabel(r'$\theta$ [°]')
    ax.set_ylabel('densità')
    ax.grid(linestyle=':')
    
    fig.show()
