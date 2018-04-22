import mc
import lab
import lab4
import numpy as np

def rutherford(theta):
    return 1 / (1 - np.cos(np.radians(theta))) ** 2

def y(x, xmin):
    return x + np.sign(x) * xmin * np.exp(-np.abs(x) / xmin)

def gauss(x, sigma):
    return np.exp(-(x / sigma) ** 2)

def function(x, ampl, xmin, sigma_xmin, gauss_ruth):
    return ampl * (rutherford(y(x, xmin)) + gauss_ruth * gauss(x, sigma_xmin * xmin))

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    
    fig = plt.figure('empirical')
    fig.clf()
    fig.set_tight_layout(True)

    ax = fig.add_subplot(111)
    
    t, w, e = mc.mc_cached(seed=0, N=100000, **mc.target_al8, **mc.coll_5)
    t = np.degrees(t)
    
    counts, edges, unc_counts = lab4.histogram(t, weights=w, bins=int(np.sqrt(len(t))))

    theta = edges[:-1] + (edges[1] - edges[0]) / 2
    cut = (np.abs(theta) >= 20) & (np.abs(theta) <= 70)
    theta = theta[cut]
    counts = counts[cut]
    unc_counts = unc_counts[cut]
    
    p0 = (10, 10, 1/2, 100000)
    out = lab.fit_curve(function, theta, counts, dy=unc_counts, p0=p0, print_info=1)

    # theta = np.linspace(-85, 85, 1000)
    ax.errorbar(theta, counts, yerr=unc_counts, fmt=',', label='mc')
    if not out.success:
        ax.semilogy(theta, function(theta, *p0), '-', linewidth=4, label='p0')
        ax.plot(theta, p0[0] * rutherford(y(theta, p0[1])), '-', label='fit_rutherford', scaley=False)
        ax.plot(theta, p0[0] * p0[3] * gauss(theta, p0[1] * p0[2]), '-', label='fit_gauss', scaley=False)
    else:
        ax.semilogy(theta, function(theta, *out.par), '-', linewidth=2, label='fit')
        ax.plot(theta, out.par[0] * rutherford(y(theta, out.par[1])), '-', label='fit_rutherford', scaley=False)
        ax.plot(theta, out.par[0] * out.par[3] * gauss(theta, out.par[1] * out.par[2]), '-', scaley=False, label='fit_gauss')
    
    ax.legend(loc=1)
    ax.grid(linestyle=':')
    
    fig.show()
