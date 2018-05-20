import numpy as np
import lsqfit
import gvar
import fit_peak
from matplotlib import pyplot as plt
import lab4

fig = plt.figure('mass18-peaks')
fig.clf()
fig.set_tight_layout(True)
axs = fig.subplots(3, 1)

figcal = plt.figure('mass18-cal')
figcal.clf()
figcal.set_tight_layout(True)
axcal = figcal.add_subplot(111)

cut = dict(
    ch1=dict(
        # left, right
        nabeta=[180, 210],
        cs=[220, 270],
        conagamma=[400, 530],
        nabetagamma=[650, 710],
        coco=[900, 980]
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
        coco=[850, 900]
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
    ax = axs[channel - 1]
    
    # load data
    filename = [
        '../DAQ/0517_nacocs_ch1_27db_2.txt',
        '../DAQ/0518_nacocs_ch2_26db.txt',
        '../DAQ/0518_nacocs_ch3_17db.txt',
    ][channel - 1]
    samples, = lab4.loadtxt(filename, usecols=(0,), unpack=True, dtype='uint16')
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
    if kw['print_info']:
        print('_________{:s}_________'.format(label))

    # na
    cut_margins = cut[label]['nabeta']
    cut_bool = (cut_margins[0] <= x) & (x <= cut_margins[1]) & cut5
    outputs, inputs = fit_peak.fit_peak(bins, hist, cut=cut_bool, bkg='exp', **kw)
    peaks['nabeta'] = outputs['peak1_mean']
    input_var['{:s}_nabeta'.format(label)] = inputs['data'] 
    
    # cs
    cut_margins = cut[label]['cs']
    cut_bool = (cut_margins[0] <= x) & (x <= cut_margins[1]) & cut5
    outputs, inputs = fit_peak.fit_peak(bins, hist, cut=cut_bool, bkg='exp', **kw)
    peaks['cs'] = outputs['peak1_mean']
    input_var['{:s}_cs'.format(label)] = inputs['data']
    
    # conagamma
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
    cut_margins = cut[label]['nabetagamma']
    cut_bool = (cut_margins[0] <= x) & (x <= cut_margins[1]) & cut5
    outputs, inputs = fit_peak.fit_peak(bins, hist, cut=cut_bool, bkg='exp', **kw)
    peaks['nabetagamma'] = outputs['peak1_mean']
    input_var['{:s}_nabetagamma'.format(label)] = inputs['data']
    
    # coco
    cut_margins = cut[label]['coco']
    cut_bool = (cut_margins[0] <= x) & (x <= cut_margins[1]) & cut5
    outputs, inputs = fit_peak.fit_peak(bins, hist, cut=cut_bool, bkg='exp', **kw)
    peaks['coco'] = outputs['peak1_mean']
    input_var['{:s}_coco'.format(label)] = inputs['data']    
    
    ax.legend(loc='upper right', fontsize='small')
    ax.grid(linestyle=':')
        
    # fit calibration and mass
    print('__________{:s} calibration and mass fit__________'.format(label))
    Y = peaks
    X = dict(
        cs=661.7,
        co117=1173.2,
        nagamma=1274.5,
        co133=1332.5,
        coco=1332.5 + 1173.2
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
        ans['nabetagamma'] = m * (p['mass'] + x['nagamma']) + q
        return ans
    fit = lsqfit.nonlinear_fit(data=(X, Y), fcn=fcn, p0=p0, debug=True)
    print(fit.format(maxline=True))
    mass[label] = fit.p['mass']
    
    # plot
    keys = list(X.keys())
    X_plot = [X[key] for key in keys] + [mass[label], mass[label] + X['nagamma']]
    Y_plot = [Y[key] for key in keys] + [Y['nabeta'], Y['nabetagamma']]
    rt = axcal.errorbar(gvar.mean(X_plot), gvar.mean(Y_plot), xerr=gvar.sdev(X_plot), yerr=gvar.sdev(Y_plot), fmt='.', label=label)
    color = rt[0].get_color()
    xspace = np.array([np.min(gvar.mean(X_plot)), np.max(gvar.mean(X_plot))])
    axcal.plot(xspace, fit.pmean['slope'] * xspace + fit.pmean['intercept'], '-', color=color)

print(gvar.fmt_errorbudget(mass, input_var))

axcal.legend(loc=0)
axcal.set_xlabel('valore nominale / fittato [keV]')
axcal.set_ylabel('media del picco [canale ADC]')
axcal.grid(linestyle=':')

fig.show()
figcal.show()
