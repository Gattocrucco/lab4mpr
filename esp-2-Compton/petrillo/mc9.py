import numba
import numpy as np
from pylab import *

@numba.jit(numba.float64[3](numba.float64[3], numba.float64[3]), nopython=True, cache=True)
def cross(a, b):
    return np.array([
        a[1]*b[2] - a[2]*b[1],
        a[2]*b[0] - a[0]*b[2],
        a[0]*b[1] - a[1]*b[0]
    ])

@numba.jit(nopython=True, cache=True)
def mc(energy, theta_0=0, N=1000, seed=0, sigma=0, center=0):
    L = 20
    R = 2
    m_e = 0.5
    sigma *= np.pi / 180
    center *= np.pi / 180
    theta_0 *= np.pi / 180
    
    X = np.array([1., 0, 0])
    Y = np.array([0, 1., 0])
    Z = np.array([0, 0, 1.])
    z = Z * np.cos(theta_0) + X * np.sin(theta_0)
    y = Y
    x = X * np.cos(theta_0) - Z * np.sin(theta_0)
    
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
        
        cos_theta = np.random.uniform(-1, 1)
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
energy, count = mc(1.33, N=N, sigma=1, seed=1)

figure('mc9')
clf()
hist(energy, bins='sqrt', histtype='step', label='acc %.2g' % (N / count,))
legend(loc=1)
show()
        