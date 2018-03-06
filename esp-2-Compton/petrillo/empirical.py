import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import histo
import lab

def gauss(x, mu, sigma):
    return np.exp(-1/2 * (x - mu)**2 / sigma**2)

def log_gauss(x, s, scale):
    return np.where(x > 0, 1 / (x / scale) * np.exp(-1/2 * (np.log(x / scale) / s)**2), 0)

def fermi(x, x0, scale):
    return 1 / (1 + np.exp((x - x0) / scale))

def slope(x, x0, slope):
    return 1 + slope * (x - x0)

f1 = lambda e, *p: p[1] * gauss(e, p[0], p[0] * p[2])
f2 = lambda e, *p: p[1] * p[3] * fermi(e, p[0] * p[4], p[0] * p[5]) * slope(e, p[0] * p[4], 1/p[0] * p[6])
f3 = lambda e, *p: p[1] * p[7] * log_gauss(-(e - p[0] * p[8]) + p[0] * p[9], p[10] / p[9], p[0] * p[9])

def empirical_secondary(e, *p):
    return f1(e, *p) + f2(e, *p) + f3(e, *p)

class EmpiricalSecondary(object):

    def __init__(self, samples, plot=False):
        """
        samples is <secondary> output from mc9.mc
        """
        if plot:
            fig = plt.figure('empirical')
            fig.clf()

            ax = fig.add_subplot(211)
            ax_di = fig.add_subplot(212)

        counts, edges = np.histogram(samples, bins='sqrt')
        norm_factor = 1 / (np.mean(counts) * (edges[-1] - edges[0]))
        norm_counts = counts * norm_factor
        if plot:
            histo.bar_line(edges, norm_counts, ax=ax)
            x = ax.get_xlim()
            y = ax.get_ylim()

        # estimate initial parameters
        p = [np.nan, np.nan, np.nan, np.nan, 0.83, 0.04, np.nan, 0.43, 0.85, 0.3, 0.07]
        idx = np.argmax(counts[len(counts) // 2:]) + len(counts) // 2
        p[1] = norm_counts[idx] # maximum of the gaussian
        p[0] = edges[idx] # mean of the gaussian
        idx_hwhm = np.sum(norm_counts[idx:] >= p[1] / 2) + idx
        p[2] = (edges[idx_hwhm] - p[0]) / (1.17 * p[0]) # sd / mean of the gaussian
        par, _ = lab.fit_linear(edges[:len(counts) // 3], norm_counts[:len(counts) // 3])
        p[3] = (par[0] * p[0] * p[4] + par[1]) / p[1] # amplitude of f2
        p[6] = par[0] * p[0] / (p[1] * p[3]) # slope at left

        if plot:
            ax.plot(edges, f1(edges, *p), '--k', linewidth=0.5)
            ax.plot(edges, f2(edges, *p), '--k', linewidth=0.5)
            ax.plot(edges, f3(edges, *p), '--k', linewidth=0.5)
            ax.plot(edges, empirical_secondary(edges, *p), '-k')
            ax.set_xlim(x)
            ax.set_ylim(y)
            
            ax_di.plot(edges[:-1], norm_counts - empirical_secondary(edges[:-1], *p), '-k')

        out = lab.fit_curve(empirical_secondary, edges[:-1] + (edges[1] - edges[0]) / 2, norm_counts, dy=np.where(counts > 0, np.sqrt(counts), 1) * norm_factor, p0=p, print_info=plot)
        
        if plot:
            ax.plot(edges, f1(edges, *out.par), '--r', linewidth=0.5)
            ax.plot(edges, f2(edges, *out.par), '--r', linewidth=0.5)
            ax.plot(edges, f3(edges, *out.par), '--r', linewidth=0.5)
            ax.plot(edges, empirical_secondary(edges, *out.par), '-r')

            ax_di.plot(edges[:-1], norm_counts - empirical_secondary(edges[:-1], *out.par), '-r')

            fig.show()
        
        self._parameters = out.par
    
    def __call__(self, x, scale):
        return 1/scale * empirical_secondary(x, scale * self._parameters[0], *self._parameters[1:])

if __name__ == '__main__':
    import mc9
    
    _, samples = mc9.mc_cached(1.33, theta_0=45, N=1000000, seed=0)
    
    empirical = EmpiricalSecondary(samples, plot=True)
    
    fig = plt.figure('empirical-test')
    fig.clf()
    ax = fig.add_subplot(111)
    fx = np.linspace(np.min(samples), np.max(samples), 500)
    ax.plot(fx, empirical(fx, 1), label='scale = 1')
    ax.plot(fx * 2, empirical(fx * 2, 2), label='scale = 2')
    ax.plot(fx / 2, empirical(fx / 2, 1/2), label='scale = 1/2')
    ax.legend()
    fig.show()
    