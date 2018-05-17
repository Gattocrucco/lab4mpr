import numpy as np
import lsqfit
import gvar
import fit_peak
from matplotlib import pyplot as plt
import lab4

fig = plt.figure('mass17-peaks')
fig.clf()
fig.set_tight_layout(True)
axs = fig.subplots(3, 1)

figcal = plt.figure('mass17-cal')
figcal.clf()
figcal.set_tight_layout(True)
axcal = figcal.add_subplot(111)

cut = dict(
    ch1=dict(
        # left, right
        nabeta=[275, 325],
        conagamma=[625, 800],
        cs=[350, 425]
    ),
    ch2=dict(
        nabeta=[300, 350],
        conagamma=[625, 800],
        cs=[375, 450]
    ),
    ch3=dict(
        nabeta=[325, 400],
        conagamma=[650, 775],
        cs=[425, 480]
    )
)

minlength = 1150
rebin = 4
bins = np.arange(minlength + 1)[::rebin]
x = (bins[:-1] + bins[1:]) / 2

input_var = {}
mass = {}
for channel in [1, 2, 3]:
    label = 'ch{:d}'.format(channel)
        
    peaks = {}
    ax = axs[channel - 1][0]
    
    # load data
    filename = '../DAQ/0515_nacocs_{:s}{:s}.txt'.format(label, '_2' if channel == 1 else '')
    samples, = lab4.loadtxt(filename, usecols=(0,), unpack=True, dtype='uint16')
    hist = lab4.rebin(np.bincount(samples, minlength=minlength), rebin)
    cut5 = hist >= 5
    hist = gvar.gvar(hist, np.sqrt(hist))
    input_var['{:s}_{:s}'.format(label, source)] = hist
    
    # plot
    ax.errorbar(x, gvar.mean(hist), yerr=gvar.sdev(hist), fmt=',', label=label)
    ax.set_yscale('log')
    ax.set_ylabel('conteggio')
    if channel == 3:
        ax.set_xlabel('canale ADC')

    # fit peaks
    kw = dict(ax=ax, plot_kw=dict(scaley=False), print_info=1)
    if kw['print_info']:
        print('_________{:s}_________'.format(label))

    # na
    cut_margins = cut[label]['nabeta']
    cut_bool = (cut_margins[0] <= x) & (x <= cut_margins[1]) & cut5
    outputs, inputs = fit_peak.fit_peak(bins, hist, cut=cut_bool, bkg='exp', **kw)
    peaks['nabeta'] = outputs['peak1_mean']
    
    # co + nagamma
    cut_margins = cut[label]['conagamma']
    cut_bool = (cut_margins[0] <= x) & (x <= cut_margins[1]) & cut5
    # mean1 = cut_margins[0] + (cut_margins[1] - cut_margins[0]) * 1/3
    # mean2 = cut_margins[0] + (cut_margins[1] - cut_margins[0]) * 2/3
    # manual_p0 = {}#dict(peak1_mean=mean1, peak2_mean=mean2, peak1_sigma=10, peak2_sigma=10)
    outputs, inputs = fit_peak.fit_peak(bins, hist, npeaks=3, cut=cut_bool, bkg='exp', **kw)
    peak123 = [outputs['peak1_mean'], outputs['peak2_mean'], outputs['peak3_mean']]
    idx = np.argsort(gvar.mean(peak123))
    peaks['co117'] = peak123[idx[0]]
    peaks['nagamma'] = peak123[idx[1]]
    peaks['co133'] = peak123[idx[2]]

    # cs
    cut_margins = cut[label]['cs']
    cut_bool = (cut_margins[0] <= x) & (x <= cut_margins[1]) & cut5
    outputs, inputs = fit_peak.fit_peak(bins, hist, cut=cut_bool, bkg='exp', **kw)
    peaks['cs'] = outputs['peak1_mean']
    
    ax.legend(loc='upper right', fontsize='small')
        
    # fit calibration and mass
    print('__________{:s} calibration and mass fit__________'.format(label))
    Y = peaks
    X = dict(
        nagamma=1275,
        co117=1173,
        co133=1333,
        cs=662
    )
    p0 = dict(
        slope=1,
        intercept=1,
        mass=400
    )
    def fcn(x, p):
        m = p['slope']
        q = p['intercept']
        ans = {}
        for key in x:
            ans[key] = m * x[key] + q
        ans['nabeta'] = m * p['mass'] + q
        return ans
    fit = lsqfit.nonlinear_fit(data=(X, Y), fcn=fcn, p0=p0, debug=True)
    print(fit.format(maxline=True))
    mass[label] = fit.p['mass']
    
    # plot
    keys = list(X.keys())
    X_plot = [X[key] for key in keys] + [mass[label]]
    Y_plot = [Y[key] for key in keys] + [Y['nabeta']]
    rt = axcal.errorbar(gvar.mean(X_plot), gvar.mean(Y_plot), xerr=gvar.sdev(X_plot), yerr=gvar.sdev(Y_plot), fmt='.', label=label)
    color = rt[0].get_color()
    xspace = np.array([np.min(gvar.mean(X_plot)), np.max(gvar.mean(X_plot))])
    axcal.plot(xspace, fit.pmean['slope'] * xspace + fit.pmean['intercept'], '-', color=color)

print(gvar.fmt_errorbudget(mass, input_var))

axcal.legend(loc=0)
axcal.set_xlabel('valore nominale / fittato [keV]')
axcal.set_ylabel('media del picco [canale ADC]')

fig.show()
figcal.show()
