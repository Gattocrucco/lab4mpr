import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import histo
import lab
import sympy as sp

def histogram(a, bins=10, range=None, weights=None, density=None):
    """
    Same as numpy.histogram, apart from extra return value.
    
    Returns
    -------
    hist
    bin_edges
    unc_hist :
        Uncertainty of hist
    """
    hist, bin_edges = np.histogram(a, bins=bins, range=range, weights=weights, density=density)
    
    if weights is None and not density:
        unc_hist = np.where(hist > 0, np.sqrt(hist), 1)
    elif weights is None:
        counts, _ = np.histogram(a, bins=bins, range=range)
        unc_hist = np.where(counts > 0, np.sqrt(counts), 1) / (len(a) * np.diff(bin_edges))
    else:
        unc_hist, _ = np.histogram(a, bins=bins, range=range, weights=weights ** 2)
        unc_hist = np.where(unc_hist > 0, np.sqrt(unc_hist), 1)
        if density:
            unc_hist /= (np.sum(weights) * np.diff(bin_edges))
    
    return hist, bin_edges, unc_hist

def gauss(x, mu, sigma):
    return sp.exp(-1/2 * (x - mu)**2 / sigma**2)

def log_gauss(x, s, scale):
    x = sp.Piecewise((x, x > 0), (scale * 1e-15 * sp.exp(s), True))
    return 1 / (x / scale) * sp.exp(-1/2 * (sp.log(x / scale) / s)**2)

def fermi(x, x0, scale):
    return 1 / (1 + sp.exp((x - x0) / scale))

def slope(x, x0, slope):
    return 1 + slope * (x - x0)

f1 = lambda e, *p: p[1] * gauss(e, p[0], p[0] * p[2])
f2 = lambda e, *p: p[1] * p[3] * fermi(e, p[0] * p[4], p[0] * p[5]) * slope(e, p[0] * p[4], 1/p[0] * p[6])
f3 = lambda e, *p: p[1] * p[7] * log_gauss(-(e - p[0] * p[8]) + p[0] * p[9], p[10] / p[9], p[0] * p[9])

def empirical_secondary(e, *p):
    return f1(e, *p) + f2(e, *p) + f3(e, *p)

syms = sp.symbols('x p0:11')
_f1 = sp.lambdify(syms, f1(*syms))
_f2 = sp.lambdify(syms, f2(*syms))
_f3 = sp.lambdify(syms, f3(*syms))
_empirical_secondary = sp.lambdify(syms, empirical_secondary(*syms))

# g1 = lambda e, *p: p[1] * gauss(e, p[1], p[2])
# g2 = lambda e, *p: p[3] * gauss(e, p[4], p[5])
#
# def empirical_primary(e, *p):
#     return g1(e, *p) + g2(e, *p)
#
# syms = sp.symbols('x p0:6')
# _g1 = sp.lambdify(syms, g1(*syms))
# _g2 = sp.lambdify(syms, g2(*syms))
# _empirical_primary(syms, empirical_primary(*syms))
#
# class EmpiricalPrimary(object):
#
#     def __init__(self, samples1, weights1, samples2, weights2, plot=False, symb=False):
#         if plot:
#             fig = plt.figure('empirical-primary')
#             fig.clf()
#
#             ax = fig.add_subplot(211)
#             ax_di = fig.add_subplot(212)
#
#         counts1, edges, unc_counts1 = histogram(samples1, bins=int(np.sqrt(len(samples1))), weights=weights1)
#         counts2,     _, unc_counts2 = histogram(samples2, bins=edges, weights=weights2)
#         counts = counts_1 + counts_2
#         unc_counts = np.sqrt(unc_counts1 ** 2 + unc_counts2 ** 2)
#
#         if plot:
#             histo.bar_line(edges, counts1, ax=ax)
#             histo.bar_line(edges, counts2, ax=ax)
#             x = ax.get_xlim()
#             y = ax.get_ylim()
#
#         p = [np.max(counts1), np.mean(samples1), np.std()]

class EmpiricalSecondary(object):

    def __init__(self, samples, weights, plot=False, symb=False):
        """
        samples is <secondary_electrons> output from mc9.mc
        weights is <weights_se>
        """
        if plot:
            fig = plt.figure('empirical-secondary')
            fig.set_tight_layout(True)
            fig.clf()

            ax = fig.add_subplot(111)
            # ax_di = fig.add_subplot(212)

        counts, edges, unc_counts = histogram(samples, bins=int(np.sqrt(len(samples))), weights=weights, density=True)
        if plot:
            line, = histo.bar_line(edges, counts, ax=ax, color='black', label='monte carlo')
            # color = line.get_color()
            # histo.bar_line(edges, counts - unc_counts, ax=ax, linewidth=.5, color=color)
            # histo.bar_line(edges, counts + unc_counts, ax=ax, linewidth=.5, color=color)
            x = ax.get_xlim()
            y = ax.get_ylim()

        # estimate initial parameters
        p = [np.nan, np.nan, np.nan, np.nan, 0.83, 0.04, np.nan, 0.35, 0.85, 0.3, 0.1]
        idx = np.argmax(counts[len(counts) // 2:]) + len(counts) // 2
        p[1] = counts[idx] # maximum of the gaussian
        p[0] = edges[idx] # mean of the gaussian
        idx_hwhm = np.sum(counts[idx:] >= p[1] / 2) + idx
        p[2] = (edges[idx_hwhm] - p[0]) / (1.17 * p[0]) # sd / mean of the gaussian
        par, _ = lab.fit_linear(edges[:len(counts) // 3], counts[:len(counts) // 3])
        p[3] = (par[0] * p[0] * p[4] + par[1]) / p[1] # amplitude of f2
        p[6] = par[0] * p[0] / (p[1] * p[3]) # slope at left

        # if plot:
            # color = [0.3] * 3
            # ax.plot(edges, _f1(edges, *p), '--',  linewidth=0.5, color=color)
            # ax.plot(edges, _f2(edges, *p), '--',  linewidth=0.5, color=color)
            # ax.plot(edges, _f3(edges, *p), '--',  linewidth=0.5, color=color)
            # ax.plot(edges, _empirical_secondary(edges, *p), '-', color=color)
            # ax.set_xlim(x)
            # ax.set_ylim(y)
            
            # ax_di.plot(edges[:-1], counts - _empirical_secondary(edges[:-1], *p), '-k')

        if symb:
            model = lab.CurveModel(empirical_secondary, symb=True, npar=11)
        else:
            model = lab.CurveModel(_empirical_secondary, symb=False, npar=11)
        out = lab.fit_curve(model, edges[:-1] + (edges[1] - edges[0]) / 2, counts, dy=unc_counts, p0=p, print_info=plot, maxit=100)
        
        if plot:
            color = [0.7] * 3
            ax.plot(edges, _empirical_secondary(edges, *out.par), '-', color=color, label='fit')
            ax.plot(edges, _f1(edges, *out.par), '--',  linewidth=0.5, color=color, label='  addendi')
            ax.plot(edges, _f2(edges, *out.par), '--',  linewidth=0.5, color=color)
            ax.plot(edges, _f3(edges, *out.par), '--',  linewidth=0.5, color=color)
            
            ax.legend(loc=2)
            ax.set_xlabel('canale ADC [digit]')
            ax.set_ylabel('densità [digit$^{-1}$]')

            # ax_di.plot(edges[:-1], counts - _empirical_secondary(edges[:-1], *out.par), '-r')

            fig.show()
        
        self._parameters = out.par
        self._fun = empirical_secondary if symb else _empirical_secondary
    
    def __call__(self, x, scale):
        return 1/scale * self._fun(x, scale * self._parameters[0], *self._parameters[1:])  

if __name__ == '__main__':
    import mc9
    
    print('monte carlo...')
    theta_0 = 45
    _, samples, _, weights = mc9.mc_cached(1.33, theta_0=theta_0, N=1000000, nai_distance=40, seed=0)
    _, s4, _, w4 = mc9.mc_cached(1.33, theta_0=theta_0, N=1000000, m_e=0.4, seed=1)
    _, s6, _, w6 = mc9.mc_cached(1.33, theta_0=theta_0, N=1000000, m_e=0.6, seed=2)
    
    print('histograms...')
    c4, e4, dc4 = histogram(s4, bins=int(np.sqrt(len(s4))), weights=w4, density=True)
    c6, e6, dc6 = histogram(s6, bins=int(np.sqrt(len(s6))), weights=w6, density=True)
    c, e = np.histogram(samples, bins=int(np.sqrt(len(samples))), weights=weights, density=True)
    
    print('empirical...')
    symb = True
    empirical = EmpiricalSecondary(samples, weights, plot=True, symb=symb)
        
    print('fit...')
    model = lab.CurveModel(lambda e, s: empirical(e, s), symb=symb)
    cut4 = e4[:-1] > 1500
    cut6 = e6[:-1] > 1500
    out4 = lab.fit_curve(model, (e4[:-1] + (e4[1] - e4[0]) / 2)[cut4], c4[cut4], dy=dc4[cut4], p0=1, print_info=1)
    out6 = lab.fit_curve(model, (e6[:-1] + (e6[1] - e6[0]) / 2)[cut6], c6[cut6], dy=dc6[cut6], p0=1, print_info=1)
    
    print('plot...')
    fig = plt.figure('empirical-test')
    fig.clf()
    fig.set_tight_layout(True)
    ax = fig.add_subplot(111)
    # fx = np.linspace(np.min(samples), np.max(samples), 500)
    fx4 = np.linspace(np.min(e4[:-1][cut4]), np.max(e4[:-1][cut4]), 500)
    fx6 = np.linspace(np.min(e6[:-1][cut6]), np.max(e6[:-1][cut6]), 500)
    # histo.bar_line(e, c, ax=ax, label='$m_e=511$ keV')
    histo.bar_line(e4, c4, ax=ax, label='$m_e=400$ keV', color=[0.8] * 3)
    histo.bar_line(e6, c6, ax=ax, label='$m_e=600$ keV', color=[0.0] * 3)
    # ax.plot(fx, model.f()(fx, 1), label='scala = 1.00')
    ax.plot(fx4, model.f()(fx4, out4.par[0]), ':', label='scala = {:.3f}'.format(out4.par[0]), color=[0.0] * 3)
    ax.plot(fx6, model.f()(fx6, out6.par[0]), '--' , label='scala = {:.3f}'.format(out6.par[0]), color=[0.7] * 3)
    ax.legend()
    ax.set_xlabel('canale ADC [digit]')
    ax.set_ylabel('densità [digit$^{-1}$]')
    fig.show()
    