import numpy as np
import lsqfit
import gvar
import fit_peak
from matplotlib import pyplot as plt
import lab4
import copy

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
        conagamma=[540, 645],
        nabetagamma=[710, 755],
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
    nparts = 1
    for part in range(nparts):
        part_label = 'part%d' % (part + 1)
        hist = lab4.rebin(np.bincount(samples[part * len(samples) // nparts:(part + 1) * len(samples) // nparts], minlength=minlength), rebin)
        cut5 = hist >= 5
        hist = gvar.gvar(hist, np.sqrt(hist))
    
        # plot
        non_zero = gvar.mean(hist) > 0
        ax.errorbar(x[non_zero], gvar.mean(hist[non_zero]), yerr=gvar.sdev(hist[non_zero]), fmt=',', label=label + ' ' + part_label)
        ax.errorbar(x[~non_zero], gvar.mean(hist[~non_zero]), yerr=[np.zeros(np.sum(~non_zero)), np.ones(np.sum(~non_zero))], fmt=',', label=label + ' ' + part_label)
        ax.set_yscale('symlog', linthreshy=1, linscaley=0.3)
        ax.set_ylabel('conteggio')
        if channel == 3:
            ax.set_xlabel('canale ADC')
    
        # fit peaks
        kw = dict(ax=ax, plot_kw=dict(scaley=False), print_info=1)
        for key in cut[label].keys():
            if kw['print_info']:
                print('_________{:s} {} {}_________'.format(label, part_label, key))
            cut_margins = cut[label][key]
            cut_bool = (cut_margins[0] <= x) & (x <= cut_margins[1]) & cut5
            npeaks = 3 if key == 'conagamma' else 1
            outputs, inputs = fit_peak.fit_peak(bins, hist, npeaks=npeaks, cut=cut_bool, bkg='exp', **kw)
            if key == 'conagamma':
                peak123_mean = np.array([outputs['peak%d_mean' % i] for i in range(1, 4)])
                peak123_norm = np.array([outputs['peak%d_norm' % i] for i in range(1, 4)])
                idx_norm = np.argsort(gvar.mean(peak123_norm))
                peak_co_mean = peak123_mean[idx_norm][1:]
                idx_mean = np.argsort(gvar.mean(peak_co_mean))
                peaks['co117'] = peak_co_mean[idx_mean[0]]
                peaks['co133'] = peak_co_mean[idx_mean[1]]
            else:
                peaks[key] = outputs['peak1_mean']
            input_var['{:s}_{}_{}'.format(label, part_label, key)] = inputs['data'] 
        
        ax.legend(loc='upper right', fontsize='small')
        ax.grid(linestyle=':')
    
        # fit calibration and mass
        Y = peaks
        X = dict(
            cs=661.7,
            co117=1173.2,
            # nagamma=1274.5,
            co133=1332.5,
            coco=1332.5 + 1173.2
        )
        p0 = dict(
            slope=1,
            intercept=1,
            mass=400,
            curvature=1,
            # cippa=1,
            # lippa=1
        )
    
        def curve(x, p):
            c = p.get('curvature', 0) / 1e6
            k = p.get('cippa', 0) / 1e9
            u = p.get('lippa', 0) / 1e12
            m = p['slope']
            q = p['intercept']
            return 24 * u * x ** 4 + 6 * k * x ** 3 + 2 * c * x ** 2 + m * x + q

        def fcn(x, p):
            ans = {}
            for key in x:
                ans[key] = curve(x[key], p)
            ans['nabeta'] = curve(p['mass'], p)
            ans['nabetagamma'] = curve(p['mass'] + 1274.5, p)
            return ans
    
        print('__________{:s} {} calibration and mass fit__________'.format(label, part_label))
        fit = lsqfit.nonlinear_fit(data=(X, Y), fcn=fcn, p0=p0, debug=True)
        mass['{}_{}'.format(label, part_label)] = fit.p['mass']
        
        chi2_dof = fit.chi2 / fit.dof
        if chi2_dof > 1:
            chi2 = gvar.gvar(0, fit.p['mass'].sdev * np.sqrt(fit.chi2 / fit.dof - 1))
            mass['{}_{}'.format(label, part_label)] += chi2
            input_var['{}_{}_chi2'.format(label, part_label)] = chi2
    
        print(fit.format(maxline=True))
    
        # plot
        keys = list(X.keys())
        X_plot = [X[key] for key in keys]
        Y_plot = [Y[key] for key in keys]
        rt = axcal.errorbar(gvar.mean(X_plot), gvar.mean(Y_plot), xerr=gvar.sdev(X_plot), yerr=gvar.sdev(Y_plot), fmt='.', label='{}_{}'.format(label, part_label))
        color = rt[0].get_color()
        X_plot += [fit.p['mass'], fit.p['mass'] + 1274.5]
        Y_plot += [Y['nabeta'], Y['nabetagamma']]
        axcal.errorbar(fit.pmean['mass'], gvar.mean(Y['nabeta']), xerr=fit.psdev['mass'], yerr=gvar.sdev(Y['nabeta']), fmt='x', color=color)
        axcal.errorbar(fit.pmean['mass'] + 1274.5, gvar.mean(Y['nabetagamma']), xerr=fit.psdev['mass'], yerr=gvar.sdev(Y['nabetagamma']), fmt='x', color=color)
        xspace = np.linspace(np.min(gvar.mean(X_plot)), np.max(gvar.mean(X_plot)), 500)
        axcal.plot(xspace, curve(xspace, fit.pmean), '-', color=color)
    
# print(gvar.fmt_errorbudget(mass, input_var))
print(gvar.tabulate(mass))

axcal.legend(loc=0)
axcal.set_xlabel('valore nominale / fittato [keV]')
axcal.set_ylabel('media del picco [canale ADC]')
axcal.grid(linestyle=':')

fig.show()
figcal.show()
