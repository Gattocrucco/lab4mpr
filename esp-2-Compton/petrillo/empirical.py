import numpy as np
import mc9
import os
from scipy import stats
import matplotlib.pyplot as plt
import histo
import lab

if not os.path.exists('empirical.npz'):
    print('running mc9...')
    kw = dict(beam_sigma=2, nai_distance=40, nai_radius=2.54, N=1000000)
    p1, s1 = mc9.mc_cal(1.33, theta_0=45, seed=1, **kw)
    p2, s2 = mc9.mc_cal(1.33, theta_0=45, m_e=0.4, **kw)
    p3, s3 = mc9.mc_cal(1.33, theta_0=45, m_e=0.6, **kw)
    print('saving in empirical.npz...')
    primary = [p1, p2, p3]
    secondary = [s1, s2, s3]
    np.savez('empirical.npz', primary=primary, secondary=secondary)
else:
    print('loading empirical.npz...')
    data = np.load('empirical.npz')
    primary = data['primary']
    secondary = data['secondary']

def gauss(x, mu, sigma):
    return np.exp(-1/2 * (x - mu)**2 / sigma**2)

def log_gauss(x, s, scale):
    return np.where(x > 0, 1 / (x / scale) * np.exp(-1/2 * (np.log(x / scale) / s)**2), 0)

def fermi(x, x0, scale):
    return 1 / (1 + np.exp((x - x0) / scale))

def slope(x, x0, slope):
    return 1 + slope * (x - x0)

def empirical_secondary(e, *p):
    amplitude = p[0]
    f1 = gauss(e, p[1], p[2])
    f2 = p[3] * fermi(e, p[4], p[5]) * slope(e, p[4], p[6])
    f3 = p[7] * log_gauss(-(e - p[8]) + p[9], p[10] / p[9], p[9])
    return amplitude * (f1 + f2 + f3)

fig = plt.figure('empirical')
fig.clf()

ax = fig.add_subplot(211)
ax_di = fig.add_subplot(212)

for i in range(len(secondary)):
    counts, edges = np.histogram(secondary[i], bins='sqrt')
    norm_factor = 1 / (np.mean(counts) * (edges[-1] - edges[0]))
    norm_counts = counts * norm_factor
    histo.bar_line(edges, norm_counts, ax=ax, label='{}'.format(i))
    if i != 0: continue
    p = (2.55, 0.53, 0.035, 0.49, 0.44, 0.02, -1, 0.43, 0.45, 0.15, 0.04)
    ax.plot(edges, p[0] * gauss(edges, p[1], p[2]), '--k')
    ax.plot(edges, p[0] * p[3] * fermi(edges, p[4], p[5]) * slope(edges, p[4], p[6]), '--k')
    ax.plot(edges, p[0] * p[7] * log_gauss(-(edges - p[8]) + p[9], p[10] / p[9], p[9]), '--k')
    ax.plot(edges, empirical_secondary(edges, *p), '-k')

    ax_di.plot(edges[:-1], norm_counts - empirical_secondary(edges[:-1], *p), '-k')

    out = lab.fit_curve(empirical_secondary, edges[:-1] + (edges[1] - edges[0]) / 2, norm_counts, dy=np.where(counts > 0, np.sqrt(counts), 1) * norm_factor, p0=p, print_info=1)

    ax.plot(edges, out.par[0] * gauss(edges, out.par[1], out.par[2]), '--r')
    ax.plot(edges, out.par[0] * out.par[3] * fermi(edges, out.par[4], out.par[5]) * slope(edges, out.par[4], out.par[6]), '--r')
    ax.plot(edges, out.par[0] * out.par[7] * log_gauss(-(edges - out.par[8]) + out.par[9], out.par[10] / out.par[9], out.par[9]), '--r')
    ax.plot(edges, empirical_secondary(edges, *out.par), '-r')

    ax_di.plot(edges[:-1], norm_counts - empirical_secondary(edges[:-1], *out.par), '-r')

fig.show()
