import numba as nb
import numpy as np
from pylab import *
from scipy import optimize

m_e = 0.511 # electron mass [MeV]

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

@nb.jit(nb.float64(nb.float64, nb.float64), nopython=True, cache=True)
def klein_nishina(E, cos_theta):
    """
    E = photon energy [MeV]
    cos_theta = cosine of polar angle
    
    from https://en.wikipedia.org/wiki/Klein–Nishina_formula
    """
    P = 1 / (1 + E / m_e * (1 - cos_theta))
    return 1/2 * P**2 * (P + P**-1 - (1 - cos_theta**2))

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

@nb.jit(nopython=True, cache=True)
def mc(energy, theta_0=0, N=1000, seed=0, sigma=0, center=0, L=20, R=2):
    sigma *= np.pi / 180
    center *= np.pi / 180
    theta_0 *= np.pi / 180
    
    X = np.array([1., 0, 0])
    Y = np.array([0, 1., 0])
    Z = np.array([0, 0, 1.])
    z = Z * np.cos(theta_0) + X * np.sin(theta_0)
    y = Y
    x = X * np.cos(theta_0) - Z * np.sin(theta_0)
    
    kn_max = klein_nishina(energy, 1)

    np.random.seed(seed)
    out_energy = np.empty(N)
    i = 0
    count = 0
    while i < N:
        theta_f = np.random.normal(loc=center, scale=sigma)
        phi_f = np.random.uniform(0, 2 * np.pi)
        
        z_f = Z * np.cos(theta_f) + np.sin(theta_f) * (X * np.cos(phi_f) + Y * np.sin(phi_f))
        y_f = cross(z_f, X)
        x_f = cross(y_f, z_f)
        
        while 1:
            cos_theta_candidate = np.random.uniform(-1, 1)
            von_neumann = np.random.uniform(0, kn_max)
            if von_neumann <= klein_nishina(energy, cos_theta_candidate):
                break
        cos_theta = cos_theta_candidate
        sin_theta = np.sqrt(1 - cos_theta**2)
        phi = np.random.uniform(0, 2 * np.pi)
        
        rho = z_f * cos_theta + sin_theta * (x_f * np.cos(phi) + y_f * np.sin(phi))
        rho_x = np.dot(rho, x)
        rho_y = np.dot(rho, y)
        rho_z = np.dot(rho, z)
        radius2 = L**2 * (rho_x**2 + rho_y**2) / rho_z**2
        
        if radius2 <= R**2 and rho_z >= 0:
            out_energy[i] = energy / (1 + energy / m_e * (1 - cos_theta))
            i += 1
        count += 1
    
    return out_energy, count

N = 1000
energy, count = mc(1.33, theta_0=45, N=N, sigma=0, seed=1)

figure('mc9')
clf()
hist(energy, bins='sqrt', histtype='step', label='acc %.2g' % (N / count,))
# theta = np.linspace(0, np.pi, 1000)
# kn = klein_nishina(0.662, theta)
# plot(np.cos(theta), kn, '-k', label='E=.662')
legend(loc=1)
show()
