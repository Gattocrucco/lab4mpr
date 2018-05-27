import numpy as np
import lsqfit
import gvar
import lab4
import matplotlib.pyplot as plt
from matplotlib import colors
from scipy import linalg

def cut(bins, cut):
    x = (bins[1:] + bins[:-1]) / 2
    
    if cut is None:
        cut = np.ones(x.shape, dtype=bool)
    elif (isinstance(cut, tuple) or isinstance(cut, list)) and len(cut) == 2:
        cut = (cut[0] <= x) & (x <= cut[1])
    
    return cut

def fit_peak(bins, hist, cut=None, npeaks=1, bkg=None, absolute_sigma=True, ax=None, print_info=False, plot_kw={}, manual_p0={}, full_output=False):
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
        NOT IMPLEMENTED
    ax : None or subplot
        Axes to plot the fit.
    print_info : boolean
        If True, print a report.
    plot_kw : dictionary
        Keyword arguments passed to all plotting functions. Overrides
        internal settings.
    manual_p0 : dictionary
        Overrides initial estimates made automatically.
    
    Returns
    -------
    outputs : dictionary
        Keys: 'peakN_norm', 'peakN_mean', 'peakN_sigma', with N starting from 1.
        Note: the normalization is by area.
    inputs : dictionary
        Keys: 'data' contains the part of the histogram selected by the cut.
        'cut' and 'chi2' contain variables to account for the uncertainty derived by
        varying the cut and for absolute_sigma=False if used (NOT IMPLEMENTED).
    """
    # data
    x = (bins[1:] + bins[:-1]) / 2
    y = hist
    
    if cut is None:
        cut = np.ones(x.shape, dtype=bool)
    elif (isinstance(cut, tuple) or isinstance(cut, list)) and len(cut) == 2:
        cut = (cut[0] <= x) & (x <= cut[1])
    
    def fun_norm(u):
        return u / (1 + gvar.exp(-2 * u)) + 1 / (2 * gvar.cosh(u))
    def fun_ampl(u):
        return u / (1 + gvar.exp(-2 * u)) + 1 / (2 * gvar.cosh(u))

    # initial parameters
    norm = np.sum(gvar.mean(y)[cut] * np.diff(bins)[cut]) / npeaks
    mean = np.linspace(np.min(x[cut]), np.max(x[cut]), npeaks + 2)[1:-1]
    sigma = (np.max(x[cut]) - np.min(x[cut])) / npeaks / 8
    
    peak_label = ['peak{:d}'.format(i + 1) for i in range(npeaks)]
    p0 = {}
    for i in range(npeaks):
        p0[peak_label[i] + '_norm_unbounded'] = norm
        p0[peak_label[i] + '_mean'] = mean[i]
        p0[peak_label[i] + '_sigma'] = sigma
    if bkg == 'exp':
        center = np.min(x[cut])
        scale = np.max(x[cut]) - np.min(x[cut])
        ampl = np.mean(gvar.mean(y[cut])) / 5
        p0['exp_ampl_unbounded'] = ampl
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
            norm = fun_norm(p[peak_label[i] + '_norm_unbounded'])
            mean = p[peak_label[i] + '_mean']
            sigma = p[peak_label[i] + '_sigma']
            ans[peak_label[i]] = norm / (np.sqrt(2 * np.pi) * sigma) * gvar.exp(-1/2 * ((x - mean) / sigma) ** 2)
        if bkg == 'exp':
            ampl = fun_ampl(p['exp_ampl_unbounded'])
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
        xspace = np.linspace(np.min(x[cut]), np.max(x[cut]), 100)
        kw = dict(label='fit{}'.format('' if success else ' (failed!)'))
        kw.update(plot_kw)
        ax.plot(xspace, fcn(xspace, fit.pmean), **kw)
        ycomp = fcn_comp(xspace, fit.pmean)
        for i in range(npeaks):
            kw = dict(linestyle='--', label=peak_label[i])
            kw.update(plot_kw)
            ax.plot(xspace, ycomp[peak_label[i]], **kw)
        if not bkg is None:
            kw = dict(linestyle=':', label='background')
            kw.update(plot_kw)
            ax.plot(xspace, ycomp['bkg'], **kw)
        if not success:
            kw = dict(linestyle='-', label='p0')
            kw.update(plot_kw)
            ax.plot(xspace, fcn(xspace, p0), **kw)
    
    # output
    inputs = dict(data=y[cut])
    outputs = {}
    for i in range(npeaks):
        norm = fun_norm(fit.p[peak_label[i] + '_norm_unbounded'])
        mean = fit.p[peak_label[i] + '_mean']
        sigma = fit.p[peak_label[i] + '_sigma']
        outputs[peak_label[i] + '_norm'] = norm
        outputs[peak_label[i] + '_mean'] = mean
        outputs[peak_label[i] + '_sigma'] = sigma
    if bkg == 'exp':
        outputs['exp_ampl'] = fun_ampl(fit.p['exp_ampl_unbounded'])
        outputs['exp_lambda'] = fit.p['exp_lambda']
    
    if full_output:
        full = dict(
            fit=fit
        )
        return outputs, inputs, full
    else:
        return outputs, inputs

def cut_2d(binsx, binsy, cutx, cuty):
    x = (binsx[1:] + binsx[:-1]) / 2
    y = (binsy[1:] + binsy[:-1]) / 2

    if cutx is None:
        cutx = np.ones(len(binsx) - 1, dtype=bool)
    elif (isinstance(cutx, tuple) or isinstance(cutx, list)) and len(cutx) == 2:
        cutx = (cutx[0] <= x) & (x <= cutx[1])

    if cuty is None:
        cuty = np.ones(len(binsy) - 1, dtype=bool)
    elif (isinstance(cuty, tuple) or isinstance(cuty, list)) and len(cuty) == 2:
        cuty = (cuty[0] <= y) & (y <= cuty[1])
    
    return np.outer(cutx, cuty)

def _gauss(x, mean, sigma):
    return 1 / (np.sqrt(2 * np.pi) * sigma) * gvar.exp(-1/2 * ((x - mean) / sigma) ** 2)

def _gauss2d(x, y, mean, sigma, corr):
    factor = 1 / (2 * np.pi * sigma[0] * sigma[1] * gvar.sqrt(1 - corr**2))
    exponent = -1/2 * 1/(1 - corr**2) * (
        ((x - mean[0]) / sigma[0]) ** 2 +
        ((y - mean[1]) / sigma[1]) ** 2 -
        2 * corr * (x - mean[0]) * (y - mean[1]) / (sigma[0] * sigma[1])
    )
    return factor * gvar.exp(exponent)
    
def fit_peak_2d(binsx, binsy, hist, cut=None, bkg=None, corr=False, ax_2d=None, ax_x=None, ax_y=None, ax_2d_diff=None, print_info=0, plot_cut=False, full_output=False):
    """
    Parameters
    ----------
    binsx : N+1 array
        Bins edges along x.
    binsy : M+1 array
        Bins edges along y.
    hist : (N, M) array of gvar
        Histogram.
    cut : None or (N, M) bool array or (cutx, cuty)
        If None, do not apply cuts. If bool array with the same
        shape as hist, it is used as hist[cut]. If tuple or list
        containing two elements cutx, cuty, each of them can be
        None, N/M bool array applying a cut along x/y, or 2-element
        tuple or list indicating lower and upper borders along x/y
        to cut.
    bkg : None or string
        'expcross' : gauss_xy + (gauss * exp)_x + (gauss * exp)_y, and
            the sigmas are the same for peak and background gaussians.
    corr : bool
        If False, the peak has principal axes fixed along x and y.
    ax_2d : None or axis
        Matplotlib axis to plot level curves of the peak.
    print_info : integer
        0: do not print, 1: short report, 2: long report.
    """
    x = (binsx[1:] + binsx[:-1]) / 2
    y = (binsy[1:] + binsy[:-1]) / 2
    xx = np.outer(x, np.ones(len(y)))
    yy = np.outer(np.ones(len(x)), y)
        
    if cut is None:
        cut = np.outer(np.ones(len(x), dtype=bool), np.ones(len(y), dtype=bool))
    elif (isinstance(cut, tuple) or isinstance(cut, list)) and len(cut) == 2:
        cutx, cuty = cut
        cut = cut_2d(binsx, binsy, cutx, cuty)
            
    xx_cut = xx[cut]
    yy_cut = yy[cut]
    hist_cut = hist[cut]
    
    def fun_norm(u):
        return u / (1 + gvar.exp(-2 * u)) + 1 / (2 * gvar.cosh(u))
    def fun_ampl(u):
        return u / (1 + gvar.exp(-2 * u)) + 1 / (2 * gvar.cosh(u))
    
    # initial parameters
    p0 = {
        'mean': np.array([np.mean(xx_cut), np.mean(yy_cut)]),
        'sigma': np.array([np.std(xx_cut), np.std(yy_cut)]) / 2,
        'norm_unbounded': np.sum(gvar.mean(hist_cut) * np.outer(np.diff(binsx), np.diff(binsy))[cut])
    }
    if corr:
        p0['corr'] = 0.01
    if bkg == 'expcross':
        center = p0['mean']
        p0['exp_lambda'] = 1 / (p0['sigma'] * 4)
        p0['exp_ampl_unbounded'] = np.array(2 * [np.max(gvar.mean(hist_cut)) / 5])
    elif bkg == 'expx':
        center = p0['mean'][0]
        p0['exp_lambda'] = 1 / (p0['sigma'][0] * 4)
        p0['exp_ampl_unbounded'] = np.max(gvar.mean(hist_cut)) / 5
    elif bkg == 'expy':
        center = p0['mean'][1]
        p0['exp_lambda'] = 1 / (p0['sigma'][1] * 4)
        p0['exp_ampl_unbounded'] = np.max(gvar.mean(hist_cut)) / 5
    elif not bkg is None:
        raise KeyError(bkg)
    
    # fit
    def fcn_comp(xxyy, p):
        xx, yy = xxyy
        
        ans = {}
        mean = p['mean']
        sigma = p['sigma']
        norm = fun_norm(p['norm_unbounded'])
        if corr:
            ans['peak'] = norm * _gauss2d(xx, yy, mean, sigma, p['corr'])
        else:
            ans['peak'] = norm * _gauss(xx, mean[0], sigma[0]) * _gauss(yy, mean[1], sigma[1])
        
        if bkg == 'expcross':
            ampl = fun_ampl(p['exp_ampl_unbounded'])
            lamda = p['exp_lambda']
            ans['bkg'] = 0
            for i in range(2):
                j = [1, 0][i]
                ans['bkg'] += ampl[i] * gvar.exp(-([xx, yy][i] - center[i]) * lamda[i]) * gvar.exp(-1/2 * (([xx, yy][j] - mean[j]) / sigma[j]) ** 2)
        elif bkg == 'expx':
            ampl = fun_ampl(p['exp_ampl_unbounded'])
            lamda = p['exp_lambda']
            ans['bkg'] = ampl * gvar.exp(-(xx - center) * lamda) * gvar.exp(-1/2 * ((yy - mean[1]) / sigma[1]) ** 2)
        elif bkg == 'expy':
            ampl = fun_ampl(p['exp_ampl_unbounded'])
            lamda = p['exp_lambda']
            ans['bkg'] = ampl * gvar.exp(-(yy - center) * lamda) * gvar.exp(-1/2 * ((xx - mean[0]) / sigma[0]) ** 2)
        else:
            ans['bkg'] = 0
        
        return ans
    
    def fcn(p):
        ans = fcn_comp((xx_cut, yy_cut), p)
        return ans['peak'] + ans['bkg']
    
    try:
        fit = lsqfit.nonlinear_fit(data=hist_cut, fcn=fcn, p0=p0, debug=True)
    except:
        # in case of error
        color = 'red'
        plot_cut = True
        p = p0
        raise
    else:
        # in case of success
        color = 'black'
        plot_cut = False or plot_cut
        p = fit.pmean
    finally:
        # plot
        if not ax_2d is None:
            # mark cut
            if plot_cut:
                ax_2d.plot(xx_cut, yy_cut, '.', color=color, markersize=2)
            # plot contours
            t = np.linspace(0, 2 * np.pi, 200)
            if corr:
                sigmax, sigmay = p['sigma']
                sigmaxy = p['corr'] * sigmax * sigmay
                C = np.array([[sigmax ** 2, sigmaxy], [sigmaxy, sigmay ** 2]])
                w, R = linalg.eigh(C)
                w = np.sqrt(w)
            else:
                w = p['sigma']
                R = np.eye(2)
            kw = dict(fill=False, edgecolor=color)
            x_base = w[0] * np.cos(t)
            y_base = w[1] * np.sin(t)
            x_base, y_base = np.einsum('ij,jk->ik', R, [x_base, y_base])
            for f in [1, 2, 3]:
                x_cont = p['mean'][0] + f * x_base
                y_cont = p['mean'][1] + f * y_base
                ax_2d.fill(x_cont, y_cont, **kw)
        if not ax_2d_diff is None:
            cut_x = np.sum(cut, axis=1, dtype=bool)
            width_x = np.sum(cut_x)
            complete_range_x = np.arange(len(cut_x))
            range_x = complete_range_x[cut_x]
            left_x = np.min(range_x)
            right_x = np.max(range_x)
            cut_x = (complete_range_x >= left_x) & (complete_range_x <= right_x)

            cut_y = np.sum(cut, axis=0, dtype=bool)
            width_y = np.sum(cut_y)
            complete_range_y = np.arange(len(cut_y))
            range_y = complete_range_y[cut_y]
            left_y = np.min(range_y)
            right_y = np.max(range_y)
            cut_y = (complete_range_y >= left_y) & (complete_range_y <= right_y)
            
            rect_cut = np.outer(cut_x, cut_y)
            ans = fcn_comp((xx[rect_cut], yy[rect_cut]), p)
            f = ans['peak'] + ans['bkg']
            hist_nan = gvar.mean(hist)
            hist_nan[~cut] = np.nan
            hist_rect = hist_nan[rect_cut].reshape(width_x, width_y)
            unc = gvar.sdev(hist[rect_cut]).reshape(width_x, width_y)
            diff = (hist_rect - f.reshape(width_x, width_y)) / unc
            im = ax_2d_diff.imshow(diff.T, extent=(binsx[left_x], binsx[right_x + 1], binsy[left_y], binsy[right_y + 1]), cmap='jet', aspect='auto', origin='lower')
            ax_2d_diff.get_figure().colorbar(im, ax=ax_2d_diff)
            # if plot_cut:
            #     ax_2d_diff.plot(xx_cut, yy_cut, '.', color=color, markersize=2)
        # plot slices
        if not ax_x is None:
            xspace = np.linspace(np.min(xx_cut), np.max(xx_cut), 200)
            yspace = np.linspace(np.min(yy_cut), np.max(yy_cut), 5)[1:-1]
            for uy in yspace:
                comp = fcn_comp((xspace, uy), p)
                line, = ax_x.plot(xspace, comp['peak'] + comp['bkg'], linestyle='-', label='y = %g' % uy)
                ax_x.plot(xspace, comp['peak'], linestyle='--', color=line.get_color())
                if not bkg is None:
                    ax_x.plot(xspace, comp['bkg'], linestyle='--', color=line.get_color())
        if not ax_y is None:
            xspace = np.linspace(np.min(xx_cut), np.max(xx_cut), 5)[1:-1]
            yspace = np.linspace(np.min(yy_cut), np.max(yy_cut), 200)
            for ux in xspace:
                comp = fcn_comp((ux, yspace), p)
                line, = ax_y.plot(yspace, comp['peak'] + comp['bkg'], linestyle='-', label='x = %g' % ux)
                ax_y.plot(yspace, comp['peak'], linestyle='--', color=line.get_color())
                if not bkg is None:
                    ax_y.plot(yspace, comp['bkg'], linestyle='--', color=line.get_color())
                    
    # report
    if print_info:
        print(fit.format(maxline=0 if print_info < 2 else True))
        
    # output
    inputs = {
        'data': hist[cut]
    }
    outputs = {
        'mean': fit.p['mean'],
        'sigma': fit.p['sigma'],
        'norm': fun_norm(fit.p['norm_unbounded'])
    }
    if corr:
        outputs.update({
            'corr': fit.p['corr']
        })
    if bkg in {'expx', 'expy', 'expcross'}:
        outputs.update({
            'exp_lambda': fit.p['exp_lambda'],
            'exp_ampl': fun_ampl(fit.p['exp_ampl_unbounded'])
        })
    if full_output:
        full = dict(
            fit=fit
        )
        return outputs, inputs, full
    else:
        return outputs, inputs

if __name__ == '__main__':
    # size = 10000
    # data = np.concatenate([np.random.normal(loc=0, scale=1, size=size), np.random.normal(loc=3, scale=0.5, size=size), np.random.exponential(scale=2, size=100000) - 5])
    # hist, edges = np.histogram(data, bins='auto')
    # hist = gvar.gvar(hist, np.sqrt(hist))
    #
    # fig = plt.figure('fit_peak')
    # fig.clf()
    # ax = fig.add_subplot(111)
    #
    # cut = gvar.mean(hist) >= 5
    # outputs, inputs = fit_peak(edges, hist, cut=cut, ax=ax, print_info=True, npeaks=2, bkg='exp')
    #
    # fig.show()
    size = 10000
    size_noise = 10000
    scalex=0.1
    scaley=1
    locx=10
    locy=10
    theta = 30
    
    theta = np.radians(theta)
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    data = np.array([
        scalex * np.random.normal(size=size),
        scaley * np.random.normal(size=size)
    ])
    data = np.einsum('ij,jk->ik', R, data) + np.array([locx, locy]).reshape(-1,1)
    range_peak = np.array([np.min(data, axis=1), np.max(data, axis=1)])
    # data = np.concatenate([data, np.array([np.random.exponential(scale=5*scale, size=size_noise) - 10*scale + locx, np.random.normal(loc=locy, scale=scale, size=size_noise)])], axis=1)
    # data = np.concatenate([data, np.array([np.random.normal(loc=locx, scale=scale, size=size_noise), np.random.exponential(scale=5*scale, size=size_noise) - 10*scale + locy])], axis=1)
    datax, datay = data
    
    fig = plt.figure('fit_peak_2d')
    fig.clf()
    ax = fig.add_subplot(222)
    ax_x = fig.add_subplot(224)
    ax_y = fig.add_subplot(221)
    
    hist, edgesx, edgesy, im = ax.hist2d(datax, datay, bins=50, cmap='gray', norm=colors.LogNorm())
    fig.colorbar(im, ax=ax)
    
    cut = cut_2d(edgesx, edgesy, list(range_peak[:,0]), list(range_peak[:,1])) & (hist >= 5)
    outputs, inputs = fit_peak_2d(edgesx, edgesy, gvar.gvar(hist, np.sqrt(hist)) / np.outer(np.diff(edgesx), np.diff(edgesy)), cut=cut, bkg=None, print_info=1, ax_2d=ax, ax_x=ax_x, ax_y=ax_y, plot_cut=True, corr=True)
    print(outputs)
    
    ax_x.legend(loc='best', fontsize='small')
    ax_y.legend(loc='best', fontsize='small')
    
    fig.show()
