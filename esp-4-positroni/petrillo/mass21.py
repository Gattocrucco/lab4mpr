import numpy as np
import lsqfit
import gvar
import fit_peak
from matplotlib import pyplot as plt
import lab4
import copy

fig = plt.figure('mass21-peaks')
fig.clf()
fig.set_tight_layout(True)
axs = fig.subplots(3, 1)

figcal = plt.figure('mass21-cal')
figcal.clf()
figcal.set_tight_layout(True)
axcal = figcal.add_subplot(111)

cut = dict(
    ch1=dict(
        # left, right
        nabeta=[280, 320],
        cs=[325, 370],
        conagamma=[470, 580],
        nabetagamma=[670, 730],
        coco=[890, 970]
    ),
    ch2=dict(
        nabeta=[330, 355],
        cs=[360, 390],
        conagamma=[470, 540],
        nabetagamma=[605, 635],
        coco=[730, 780]
    ),
    ch3=dict(
        nabeta=[305, 350],
        cs=[360, 420],
        conagamma=[540, 640],
        nabetagamma=[710, 750],
        coco=[850, 920]
    )
)

minlength = 1150
rebin = 6
bins = np.arange(minlength + 1)[::rebin]
x = (bins[:-1] + bins[1:]) / 2

input_var = {}
mass = {}
for channel in [1]:
    label = 'ch{:d}'.format(channel)
    
    peaks = {}
    ax = axs[channel - 1]
    
    # load data
    filename = [
        '../DAQ/0521_nacocs_ch1_594V_14db.txt',
        '../DAQ/0518_nacocs_ch2_26db.txt',
        '../DAQ/0518_nacocs_ch3_17db.txt',
    ][channel - 1]
    samples, = lab4.loadtxt(filename, usecols=(0,), unpack=True, dtype='uint16')
    samples = samples[:len(samples) // 2]
    hist = lab4.rebin(np.bincount(samples, minlength=minlength), rebin)
    cut5 = hist >= 5
    hist = gvar.gvar(hist, np.sqrt(hist))
    
    # plot
    ax.errorbar(x, gvar.mean(hist), yerr=gvar.sdev(hist), fmt=',', label=label)
    ax.set_yscale('log')
    ax.set_ylabel('conteggio')
    if channel == 3:
        ax.set_xlabel('canale ADC')
    
    # fit peaks
    kw = dict(ax=ax, plot_kw=dict(scaley=False), print_info=1)

    # nabeta
    if kw['print_info']:
        print('_________{:s} nabeta_________'.format(label))
    cut_margins = cut[label]['nabeta']
    cut_bool = (cut_margins[0] <= x) & (x <= cut_margins[1]) & cut5
    outputs, inputs = fit_peak.fit_peak(bins, hist, cut=cut_bool, bkg='exp', **kw)
    peaks['nabeta'] = outputs['peak1_mean']
    input_var['{:s}_nabeta'.format(label)] = inputs['data'] 
    
    # cs
    if kw['print_info']:
        print('_________{:s} cs_________'.format(label))
    cut_margins = cut[label]['cs']
    cut_bool = (cut_margins[0] <= x) & (x <= cut_margins[1]) & cut5
    outputs, inputs = fit_peak.fit_peak(bins, hist, cut=cut_bool, bkg='exp', **kw)
    peaks['cs'] = outputs['peak1_mean']
    input_var['{:s}_cs'.format(label)] = inputs['data']
    
    # conagamma
    if kw['print_info']:
        print('_________{:s} conagamma_________'.format(label))
    cut_margins = cut[label]['conagamma']
    cut_bool = (cut_margins[0] <= x) & (x <= cut_margins[1]) & cut5
    outputs, inputs = fit_peak.fit_peak(bins, hist, npeaks=3, cut=cut_bool, bkg='exp', **kw)
    peak123 = [outputs['peak1_mean'], outputs['peak2_mean'], outputs['peak3_mean']]
    idx = np.argsort(gvar.mean(peak123))
    peaks['co117'] = peak123[idx[0]]
    peaks['nagamma'] = peak123[idx[1]]
    peaks['co133'] = peak123[idx[2]]
    input_var['{:s}_conagamma'.format(label)] = inputs['data']

    # nabetagamma
    if kw['print_info']:
        print('_________{:s} nabetagamma_________'.format(label))
    cut_margins = cut[label]['nabetagamma']
    cut_bool = (cut_margins[0] <= x) & (x <= cut_margins[1]) & cut5
    outputs, inputs = fit_peak.fit_peak(bins, hist, cut=cut_bool, bkg='exp', **kw)
    peaks['nabetagamma'] = outputs['peak1_mean']
    input_var['{:s}_nabetagamma'.format(label)] = inputs['data']
    
    # coco
    if kw['print_info']:
        print('_________{:s} coco_________'.format(label))
    cut_margins = cut[label]['coco']
    cut_bool = (cut_margins[0] <= x) & (x <= cut_margins[1]) & cut5
    outputs, inputs = fit_peak.fit_peak(bins, hist, cut=cut_bool, bkg='exp', **kw)
    peaks['coco'] = outputs['peak1_mean']
    input_var['{:s}_coco'.format(label)] = inputs['data']    
    
    ax.legend(loc='upper right', fontsize='small')
    ax.grid(linestyle=':')
    
    # fit calibration and mass
    Y = peaks
    X_lower = dict(
        cs=661.7,
        co117=1173.2
    )
    X_upper = dict(
        co133=1332.5,
        coco=1332.5 + 1173.2
    )
    X = dict(nagamma=1274.5)
    X.update(X_lower)
    X.update(X_upper)
    p0 = dict(
        slope=1,
        intercept=1,
        mass=400
    )
    def filter_dict(d, **kw):
        new_d = {}
        for key in d:
            if key in kw:
                new_d[key] = d[key]
        return new_d
    def fcn(x, p):
        m = p['slope']
        q = p['intercept']
        ans = {}
        for key in x:
            ans[key] = m * x[key] + q
        if 'cs' in x:
            ans['nabeta'] = m * p['mass'] + q
        if 'coco' in x:
            ans['nabetagamma'] = m * (p['mass'] + X['nagamma']) + q
        return ans
    print('__________{:s} calibration and mass fit__________'.format(label))
    fit = lsqfit.nonlinear_fit(data=(X, Y), fcn=fcn, p0=p0, debug=True)
    chi2 = gvar.gvar(0, fit.p['mass'].sdev * np.sqrt(fit.chi2 / fit.dof - 1))
    mass[label] = fit.p['mass'] + chi2
    input_var[label + '_chi2'] = chi2
    
    print(fit.format(maxline=True))
    # print('__________{:s} lower calibration and mass fit__________'.format(label))
    # fit_lower = lsqfit.nonlinear_fit(data=(X_lower, filter_dict(Y, nabeta=0, **X_lower)), fcn=fcn, p0=p0, debug=True)
    # print(fit_lower.format())
    # print('__________{:s} upper calibration and mass fit__________'.format(label))
    # fit_upper = lsqfit.nonlinear_fit(data=(X_upper, filter_dict(Y, nabetagamma=0, **X_upper)), fcn=fcn, p0=p0, debug=True)
    # print(fit_upper.format())
    
    # mass[label + 'low'] = fit_lower.p['mass']
    # mass[label + 'up'] = fit_upper.p['mass']
    
    # plot
    keys = list(X.keys())
    X_plot = [X[key] for key in keys]
    Y_plot = [Y[key] for key in keys]
    rt = axcal.errorbar(gvar.mean(X_plot), gvar.mean(Y_plot), xerr=gvar.sdev(X_plot), yerr=gvar.sdev(Y_plot), fmt='.', label=label)
    color = rt[0].get_color()
    X_plot += [mass[label], mass[label] + X['nagamma']]
    Y_plot += [Y['nabeta'], Y['nabetagamma']]
    axcal.errorbar(fit.pmean['mass'], gvar.mean(Y['nabeta']), xerr=fit.psdev['mass'], yerr=gvar.sdev(Y['nabeta']), fmt='x', color=color)
    axcal.errorbar(fit.pmean['mass'] + X['nagamma'], gvar.mean(Y['nabetagamma']), xerr=fit.psdev['mass'], yerr=gvar.sdev(Y['nabetagamma']), fmt='x', color=color)
    xspace = np.array([np.min(gvar.mean(X_plot)), np.max(gvar.mean(X_plot))])
    axcal.plot(xspace, fit.pmean['slope'] * xspace + fit.pmean['intercept'], '-', color=color)
    
    # keys = list(X_lower.keys())
    # X_plot = [X_lower[key] for key in keys] + [fit_lower.p['mass']]
    # Y_plot = [Y[key] for key in keys] + [Y['nabeta']]
    # axcal.errorbar(fit_lower.pmean['mass'], gvar.mean(Y['nabeta']), xerr=fit_upper.psdev['mass'], yerr=gvar.sdev(Y['nabeta']), fmt='x', color=color)
    # xspace = np.array([np.min(gvar.mean(X_plot)), np.max(gvar.mean(X_plot))])
    # axcal.plot(xspace, fit_lower.pmean['slope'] * xspace + fit_lower.pmean['intercept'], ':', color=color)
    #
    # keys = list(X_upper.keys())
    # X_plot = [X_upper[key] for key in keys] + [fit_upper.p['mass'] + X['nagamma']]
    # Y_plot = [Y[key] for key in keys] + [Y['nabetagamma']]
    # xspace = np.array([np.min(gvar.mean(X_plot)), np.max(gvar.mean(X_plot))])
    # axcal.errorbar(fit_upper.pmean['mass'] + X['nagamma'], gvar.mean(Y['nabetagamma']), xerr=fit_upper.psdev['mass'], yerr=gvar.sdev(Y['nabetagamma']), fmt='x', color=color)
    # axcal.plot(xspace, fit_upper.pmean['slope'] * xspace + fit_upper.pmean['intercept'], '--', color=color)

print(gvar.fmt_errorbudget(mass, input_var))

axcal.legend(loc=0)
axcal.set_xlabel('valore nominale / fittato [keV]')
axcal.set_ylabel('media del picco [canale ADC]')
axcal.grid(linestyle=':')

fig.show()
figcal.show()
