import numpy as np
import mc9
import empirical
import matplotlib.pyplot as plt
import histo
import lab
import sympy as sp

pa, sa = mc9.mc_cached(1.33, theta_0=0, N=1000000, seed=0)
pb, sb = mc9.mc_cached(1.17, theta_0=0, N=1000000, seed=1)

edges = np.arange(2**13 + 1)

countsa = np.bincount(np.asarray(np.floor(np.concatenate([pa, sa])), dtype='u2'), minlength=2**13)[:2**13]
countsb = np.bincount(np.asarray(np.floor(np.concatenate([pb, sb])), dtype='u2'), minlength=2**13)[:2**13]
counts = np.bincount(np.asarray(np.floor(np.concatenate([pb, sb, pa, sa])), dtype='u2'), minlength=2**13)[:2**13]

empa = empirical.EmpiricalSecondary(sa)
empb = empirical.EmpiricalSecondary(sb)

def fit_fun_a(e, N1, mu1, sigma1, Ns1, scale1, N2, mu2, sigma2, Ns2, scale2):
    gaus1 = N1 / (sp.sqrt(2 * np.pi) * sigma1) * sp.exp(-1/2 * (e - mu1)**2 / sigma1**2)
    sh1 = Ns1 * empa(e, scale1)
    return gaus1 + sh1
    
def fit_fun_b(e, N1, mu1, sigma1, Ns1, scale1, N2, mu2, sigma2, Ns2, scale2):
    gaus2 = N2 / (sp.sqrt(2 * np.pi) * sigma2) * sp.exp(-1/2 * (e - mu2)**2 / sigma2**2)
    sh2 = Ns2 * empb(e, scale2)
    return gaus2 + sh2

def fit_fun(e, *p):
    return fit_fun_a(e, *p) + fit_fun_b(e, *p)

p0 = [len(pa), np.mean(pa), np.std(pa), len(sa), 1, len(pb), np.mean(pb), np.std(pb), len(sb), 1]

# for i in range(len(p0)):
#     p0[i] = np.random.uniform(0.98 * p0[i], 1.02 * p0[i])

model = lab.CurveModel(fit_fun, symb=True, npar=len(p0))
out = lab.fit_curve(model, edges[:-1] + 0.5, counts, dy=np.where(counts > 0, np.sqrt(counts), 1), p0=p0, print_info=1, method='auto')

fig = plt.figure('fit')
fig.clf()
ax = fig.add_subplot(111)

modela = lab.CurveModel(fit_fun_a, symb=True)
modelb = lab.CurveModel(fit_fun_b, symb=True)

histo.bar_line(edges, countsa, ax=ax)
histo.bar_line(edges, countsb, ax=ax)
histo.bar_line(edges, counts, ax=ax)
ax.plot(edges, model.f()(edges, *p0), '-k')
ax.plot(edges, model.f()(edges, *out.par), '-r')
ax.plot(edges, modela.f()(edges, *out.par), '--r', linewidth=0.5)
ax.plot(edges, modelb.f()(edges, *out.par), '--r', linewidth=0.5)

fig.show()
