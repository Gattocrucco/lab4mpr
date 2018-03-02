import numpy as np
import mc9
import os
from scipy import stats
import matplotlib.pyplot as plt
import histo

if not os.path.exists('empirical.npz'):
    print('running mc9...')
    primary, secondary = mc9.mc(1.33, theta_0=45, N=1000000, beam_sigma=2, nai_distance=40, seed=1)
    print('saving in empirical.npz...')
    np.savez('empirical.npz', primary=primary, secondary=secondary)
else:
    print('loading empirical.npz...')
    data = np.load('empirical.npz')
    primary = data['primary']
    secondary = data['secondary']

def gauss(x, mu, sigma):
    return np.exp(-1/2 * (x - mu)**2 / sigma**2)

def fermi(x, x0, scale):
    return 1 / (1 + np.exp((x - x0) / scale))

def parabola(x, x0, slope, curv):
    return 1 + slope * (x - x0) + curv * (x - x0)**2

def empirical_secondary(e, *p):
    return p[0] * gauss(e, p[1], p[2]) + p[3] * fermi(e, p[4], p[5])

fig = plt.figure('empirical')
fig.clf()

ax = fig.add_subplot(211)
counts, edges = np.histogram(secondary, bins='sqrt')
norm_counts = counts / (np.mean(counts) * (edges[-1] - edges[0]))
histo.bar_line(edges, norm_counts, ax=ax)

p = (2.55, 0.53, 0.035, 1.4, 0.5, 0.02)
ax.plot(edges, p[0] * gauss(edges, p[1], p[2]), '--k')
ax.plot(edges, p[3] * fermi(edges, p[4], p[5]), '--k')
ax.plot(edges, empirical_secondary(edges, *p), '-k')

ax_di = fig.add_subplot(212)
ax_di.plot(edges[:-1], norm_counts - empirical_secondary(edges[:-1], *p), '-k')

fig.show()
