import numpy as np
import lsqfit
import gvar
import lab4
import matplotlib.pyplot as plt

def fit_peak(bins, hist, cut=None, npeaks=1, bkg=None, absolute_sigma=True, ax=None, print_info=False):
    """
    Fit gaussian peaks on a histogram with background.
    
    Parameters
    ----------
    bins : array
        Bins edges.
    hist : array
        Counts in the bins.
    cut : array or None
        Boolean array selecting certain bins. Same shape as hist.
    npeaks : integer >= 1
        Number of gaussian peaks to fit.
    bkg : one of None, 'exp'
        Background type.
    absolute_sigma : boolean
        If True, ordinary fit. If False, rescale uncertainties with the chisquare.
    ax : None or subplot
        Axes to plot the fit.
    print_info : boolean
        If True, print a report.
    
    Returns
    -------
    outputs : dictionary
        Keys: 'peakN_norm', 'peakN_mean', 'peakN_sigma', with N starting from 1.
        Note: the normalization is by area, not by count, they are equivalent
        if the bin width is 1.
    inputs : dictionary
        Keys: 'data' contains the part of the histogram selected by the cut.
        'cut' and 'chi2' contain variables to account for the uncertainty derived by
        varying the cut and for absolute_sigma=False if used.
    """
    # data
    x = (bins[1:] + bins[:-1]) / 2
    y = gvar.gvar(hist, np.sqrt(hist))
    
    # initial parameters
    norm = np.sum(hist[cut] * np.diff(bins)[cut]) / npeaks
    mean = np.mean(x[cut])
    sigma = np.std(x[cut])
    
    peak_label = ['peak{:d}'.format(i + 1) for i in range(npeaks)]
    p0 = {}
    for i in range(npeaks):
        p0[peak_label[i]] = [np.log(norm), mean, sigma]
    if bkg == 'exp':
        center = mean
        p0['log_exp_ampl'] = 0
        p0['log_exp_length'] = np.log(sigma)
    elif bkg != None:
        raise KeyError(bkg)
    
    # fit
    def fcn_comp(x, p):
        ans = {}
        for i in range(npeaks):
            lognorm, mean, sigma = p[peak_label[i]]
            norm = gvar.exp(lognorm)
            ans[peak_label[i]] = norm / (np.sqrt(2 * np.pi) * sigma) * gvar.exp(-1/2 * ((x - mean) / sigma) ** 2)
        if bkg == 'exp':
            ampl = gvar.exp(p['log_exp_ampl'])
            length = gvar.exp(p['log_exp_length'])
            ans['bkg'] = ampl * gvar.exp(-(x - center) / length)
        return ans
        
    def fcn(x, p):
        return sum(list(fcn_comp(x, p).values()))
    
    if cut is None:
        cut = np.ones(x.shape, dtype=bool)
    
    fit = lsqfit.nonlinear_fit(data=(x[cut], y[cut]), p0=p0, fcn=fcn, debug=True)
    success = (fit.error is None) and (fit.stopping_criterion > 0)
    
    # report
    if print_info:
        if not success:
            print('fit failed! error message: {}'.format(fit.error))
        print(fit.format(maxline=True))
    
    # TODO absolute_sigma=False
    if not absolute_sigma:
        raise ValueError('absolute_sigma not implemented')
    # TODO cut variation
    
    # plot
    if not ax is None:
        lab4.bar(bins, hist, label='dati')
        xspace = np.linspace(np.min(x[cut]), np.max(x[cut]), 100)
        ax.plot(xspace, fcn(xspace, fit.pmean), label='fit{}'.format('' if success else ' (failed!)'))
        ycomp = fcn_comp(xspace, fit.pmean)
        for i in range(npeaks):
            ax.plot(xspace, ycomp[peak_label[i]], linestyle='--', label=peak_label[i])
        ax.plot(xspace, ycomp['bkg'], linestyle='--', label='background')
        if not success:
            ax.plot(xspace, fcn(xspace, fit.p0), label='p0')
        
        ax.set_xlabel('canale ADC')
        ax.set_ylabel('conteggio')
        ax.legend(loc='best')
    
    # output
    inputs = dict(data=y[cut])
    outputs = {}
    for i in range(npeaks):
        lognorm, mean, sigma = fit.p[peak_label[i]]
        norm = gvar.exp(lognorm)
        outputs[peak_label[i] + '_norm'] = norm
        outputs[peak_label[i] + '_mean'] = mean
        outputs[peak_label[i] + '_sigma'] = sigma
    
    return outputs, inputs

if __name__ == '__main__':
    size = 10000
    data = np.concatenate([np.random.normal(loc=0, scale=1, size=size), np.random.normal(loc=3, scale=0.5, size=size), np.random.exponential(scale=2, size=100000) - 5])
    hist, edges = np.histogram(data, bins='auto')
    
    fig = plt.figure('fit_peak')
    fig.clf()
    ax = fig.add_subplot(111)
    
    cut = hist >= 5
    outputs, inputs = fit_peak(edges, hist, cut=cut, ax=ax, print_info=True, npeaks=2, bkg='exp')
    
    fig.show()
