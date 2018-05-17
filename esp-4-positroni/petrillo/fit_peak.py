import numpy as np
import lsqfit
import gvar
import lab4
import matplotlib.pyplot as plt

def fit_peak(bins, hist, cut=None, npeaks=1, bkg=None, absolute_sigma=True, ax=None, print_info=False, plot_kw={}, manual_p0={}):
    """
    Fit gaussian peaks on a histogram with background.
    
    Parameters
    ----------
    bins : array
        Bins edges.
    hist : array of gvar
        Values in the bins.
    cut : array or None
        Boolean array selecting certain bins with same shape as hist,
        or tuple or list of 2 elements to select an interval in the bin centers.
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
        Note: the normalization is by area.
    inputs : dictionary
        Keys: 'data' contains the part of the histogram selected by the cut.
        'cut' and 'chi2' contain variables to account for the uncertainty derived by
        varying the cut and for absolute_sigma=False if used.
    """
    # data
    x = (bins[1:] + bins[:-1]) / 2
    y = hist
    
    if cut is None:
        cut = np.ones(x.shape, dtype=bool)
    elif (isinstance(cut, tuple) or isinstance(cut, list)) and len(cut) == 2:
        cut = (cut[0] <= x) & (x <= cut[1])
    
    # initial parameters
    norm = np.sum(gvar.mean(y)[cut] * np.diff(bins)[cut]) / npeaks
    mean = np.linspace(np.min(x[cut]), np.max(x[cut]), npeaks + 2)[1:-1]
    sigma = (np.max(x[cut]) - np.min(x[cut])) / npeaks / 8
    
    peak_label = ['peak{:d}'.format(i + 1) for i in range(npeaks)]
    p0 = {}
    for i in range(npeaks):
        p0[peak_label[i] + '_lognorm'] = np.log(norm)
        p0[peak_label[i] + '_mean'] = mean[i]
        p0[peak_label[i] + '_sigma'] = sigma
    if bkg == 'exp':
        center = np.min(x[cut])
        scale = np.max(x[cut]) - np.min(x[cut])
        ampl = np.mean(gvar.mean(y[cut])) / 5
        p0['log_exp_ampl'] = np.log(ampl)
        p0['exp_lambda'] = 1 / scale
    elif bkg != None:
        raise KeyError(bkg)
    
    for key in manual_p0:
        if key in p0:
            p0[key] = manual_p0[key]
    
    # fit
    def fcn_comp(x, p):
        ans = {}
        for i in range(npeaks):
            norm = gvar.exp(p[peak_label[i] + '_lognorm'])
            mean = p[peak_label[i] + '_mean']
            sigma = p[peak_label[i] + '_sigma']
            ans[peak_label[i]] = norm / (np.sqrt(2 * np.pi) * sigma) * gvar.exp(-1/2 * ((x - mean) / sigma) ** 2)
        if bkg == 'exp':
            ampl = gvar.exp(p['log_exp_ampl'])
            lamda = p['exp_lambda']
            ans['bkg'] = ampl * gvar.exp(-(x - center) * lamda)
        return ans
        
    def fcn(x, p):
        return sum(list(fcn_comp(x, p).values()))
    
    try:
        fit = lsqfit.nonlinear_fit(data=(x[cut], y[cut]), p0=p0, fcn=fcn, debug=False)
    except:
        if not ax is None:
            xspace = np.linspace(np.min(x[cut]), np.max(x[cut]), 100)
            ax.plot(xspace, fcn(xspace, p0), label='p0', **plot_kw)
        raise
    success = (fit.error is None) and (fit.stopping_criterion > 0)        
    
    # report
    if print_info:
        if not success:
            print('fit failed!'.format(fit.error))
        print(fit.format())
    
    # TODO absolute_sigma=False
    if not absolute_sigma:
        raise ValueError('absolute_sigma not implemented')
    # TODO cut variation
    
    # plot
    if not ax is None:
        # lab4.bar(bins, gvar.mean(y), label='dati', ax=ax, **plot_kw)
        xspace = np.linspace(np.min(x[cut]), np.max(x[cut]), 100)
        ax.plot(xspace, fcn(xspace, fit.pmean), label='fit{}'.format('' if success else ' (failed!)'), **plot_kw)
        ycomp = fcn_comp(xspace, fit.pmean)
        for i in range(npeaks):
            ax.plot(xspace, ycomp[peak_label[i]], linestyle='--', label=peak_label[i], **plot_kw)
        if not bkg is None:
            ax.plot(xspace, ycomp['bkg'], linestyle='--', label='background', **plot_kw)
        if not success:
            pass
            ax.plot(xspace, fcn(xspace, p0), linestyle='-', label='p0', **plot_kw)
    
    # output
    inputs = dict(data=y[cut])
    outputs = {}
    for i in range(npeaks):
        norm = gvar.exp(fit.p[peak_label[i] + '_lognorm'])
        mean = fit.p[peak_label[i] + '_mean']
        sigma = fit.p[peak_label[i] + '_sigma']
        outputs[peak_label[i] + '_norm'] = norm
        outputs[peak_label[i] + '_mean'] = mean
        outputs[peak_label[i] + '_sigma'] = sigma
    
    return outputs, inputs

if __name__ == '__main__':
    size = 10000
    data = np.concatenate([np.random.normal(loc=0, scale=1, size=size), np.random.normal(loc=3, scale=0.5, size=size), np.random.exponential(scale=2, size=100000) - 5])
    hist, edges = np.histogram(data, bins='auto')
    hist = gvar.gvar(hist, np.sqrt(hist))
    
    fig = plt.figure('fit_peak')
    fig.clf()
    ax = fig.add_subplot(111)
    
    cut = gvar.mean(hist) >= 5
    outputs, inputs = fit_peak(edges, hist, cut=cut, ax=ax, print_info=True, npeaks=2, bkg='exp')
    
    fig.show()
