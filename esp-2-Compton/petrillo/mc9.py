from pylab import *
import numba
import numpy as np

E = [ 0.059,
      0.662,
     0.546 ,
     1.275 ,
     1.177 ,
     1.332 ]

sigma = [20.956, 108.68, 95.9617, 139.34, 136.0, 159.6]

dE = [6.81399117667e-05, 0.00016423, 0.0001412397, 0.0005063984, 0.000221, 0.00026697890]

dsigma = [0.525017422696, 1.86253608472, 1.10475147149, 4.91607907266, 1.3, 1.7]

f_sigma = lambda E: (2.27 + 7.28 * E ** -0.29 - 2.41 * E ** 0.21) * E / (100 * 2.35)

figure('mc9')
clf()
f_E = linspace(min(E), max(E), 500)
plot(f_E, f_sigma(f_E) / 1.81e-4, '-r')
errorbar(E, sigma, xerr=dE, yerr=dsigma, fmt=',k')
show()

@numba.jit(nopython=True)
def mc(angle=0, N=10000, sigma=True, L=20, r=2):
    out_energy = np.empty(N)
    theta_l = angle - np.arctan(r/L)
    theta_r = angle + np.arctan(r/L)
    
    for i in range(len(out_energy)):
        
