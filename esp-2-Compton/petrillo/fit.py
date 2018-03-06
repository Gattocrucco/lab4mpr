import numpy as np
import mc9
import empirical
import matplotlib.pyplot as plt
import histo
import lab

pa, sa = mc9.mc_cached(1.33, theta_0=10, N=1000000, seed=0)
pb, sb = mc9.mc_cached(1.17, theta_0=10, N=1000000, seed=1)

edges = np.arange(2**13 + 1)

countsa = np.bincount(np.asarray(np.floor(np.concatenate([pa, sa])), dtype='u2'), minlength=2**13)[:2**13]
countsb = np.bincount(np.asarray(np.floor(np.concatenate([pb, sb])), dtype='u2'), minlength=2**13)[:2**13]
counts = np.bincount(np.asarray(np.floor(np.concatenate([pb, sb, pa, sa])), dtype='u2'), minlength=2**13)[:2**13]

empa = empirical.EmpiricalSecondary(sa)
empb = empirical.EmpiricalSecondary(sb)

def fit_fun(e, N1, mu1, sigma1, Ns1, scale1, N2, mu2, sigma2, Ns2, scale2):
    gaus1 = N1 / (np.sqrt(2 * np.pi) * sigma1) * np.exp(-1/2 * (e - mu1)**2 / sigma1**2)
    sh1 = Ns1 * empa(e, scale1)
    gaus2 = N2 / (np.sqrt(2 * np.pi) * sigma2) * np.exp(-1/2 * (e - mu2)**2 / sigma2**2)
    sh2 = Ns2 * empb(e, scale2)
    return gaus1 + gaus2 + sh1 + sh2

p0 = [len(pa), np.mean(pa), np.std(pa), len(sa), 1, len(pb), np.mean(pb), np.std(pb), len(sb), 1]

for i in range(len(p0)):
    p0[i] = np.random.uniform(0.9 * p0[i], 1.1 * p0[i])

out = lab.fit_curve(fit_fun, edges[:-1] + 0.5, counts, dy=np.where(counts > 0, np.sqrt(counts), 1), p0=p0, print_info=1, method='leastsq')

fig = plt.figure('fit')
fig.clf()
ax = fig.add_subplot(111)

histo.bar_line(edges, countsa, ax=ax)
histo.bar_line(edges, countsb, ax=ax)
histo.bar_line(edges, counts, ax=ax)
ax.plot(edges, fit_fun(edges, *p0), '-k')
ax.plot(edges, fit_fun(edges, *out.par), '-r')

fig.show()
