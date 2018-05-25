import numpy as np
import lsqfit
import gvar
import fit_peak
from matplotlib import pyplot as plt
import lab4
import copy
import lab

print_peak_fit = False

def change_error(x, factor=1, corr=False):
    if corr:
        delta = (x - gvar.mean(x)) * (factor - 1)
    else:
        assert factor >= 1
        delta = gvar.gvar(0, gvar.sdev(x) * np.sqrt(factor**2 - 1))
    return delta
                
fig = plt.figure('mass18-peaks', figsize=[6.87, 6.49])
fig.clf()
fig.set_tight_layout(True)
axs = fig.subplots(3, 1)

figcal = plt.figure('mass18-cal', figsize=[4.44, 3.51])
figcal.clf()
figcal.set_tight_layout(True)
axcal = figcal.add_subplot(111)

peak_labels = dict(
    nabeta='Na$_{\\beta}$',
    cs='Cs',
    co117='Co$_{1.17}$, Co$_{1.33}$',
    # nagamma='Na$_{\\gamma}$',
    # co133='Co$_{1.33}$',
    nabetagamma='Na$_{\\beta+\\gamma}$',
    coco='Co$_{1.17+1.33}$'
)

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

rebin = 4

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
        hist = np.bincount(samples[part * len(samples) // nparts:(part + 1) * len(samples) // nparts])
        bins = np.arange(len(hist) + 1)[::rebin]
        x = (bins[:-1] + bins[1:]) / 2
        hist = lab4.rebin(hist, rebin)
        cut5 = hist >= 5
        hist = gvar.gvar(hist, np.sqrt(hist))
    
        # plot
        if nparts == 1:
            kw = dict(color='gray')
        else:
            kw = dict()
        non_zero = gvar.mean(hist) > 0
        if nparts == 1:
            data_label = 'dati PMT %d' % channel
        else:
            data_label = label + ' ' + part_label
        rt = ax.errorbar(x[non_zero], gvar.mean(hist[non_zero]), yerr=gvar.sdev(hist[non_zero]), fmt=',', label=data_label, **kw)
        ax.errorbar(x[~non_zero], gvar.mean(hist[~non_zero]), yerr=[np.zeros(np.sum(~non_zero)), np.ones(np.sum(~non_zero))], fmt=',', color=rt[0].get_color())
        if part == 0:
            ax.set_yscale('symlog', linthreshy=1, linscaley=0.3)
            if rebin == 1:
                ylabel = 'conteggio'
            else:
                ylabel = 'conteggio / %d canali' % rebin
            ax.set_ylabel(ylabel)
            if channel == 3:
                ax.set_xlabel('canale ADC')
            if channel == 1:
                ax.plot([1e6], [1e6], '-k', label='fit (gaussiana + fondo)', scalex=False, scaley=False)
                ax.plot([1e6], [1e6], '--k', label='fit (gaussiana)', scalex=False, scaley=False)
                ax.plot([1e6], [1e6], ':k', label='fit (fondo)', scalex=False, scaley=False)
        if nparts == 1:
            ax.set_xlim(*[(-218, 1020), (304, 824), (252, 976)][channel - 1])
            ax.set_ylim(*[(-0.467, 11300), (-0.866, 44900), (-0.738, 18600)][channel - 1])
    
        # fit peaks
        kw = dict(ax=ax, plot_kw=dict(scaley=False, scalex=False, color='black', label=None), print_info=print_peak_fit)
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
        
        print('_________{:s} {} peaks_________'.format(label, part_label))
        print(lab.format_par_cov(gvar.mean(peaks), gvar.evalcov(peaks)))
        
        for key in peak_labels:
            text_x = gvar.mean(peaks[key])
            h = gvar.mean(hist)
            text_y = h[x <= text_x][-1]
            if key in ['nabeta', 'cs']:
                text_y /= 1000
            else:
                text_y *= 2
            if key == 'co117':
                text_x = (gvar.mean(peaks[key]) + gvar.mean(peaks['co133'])) / 2
            ax.text(text_x, text_y, peak_labels[key], horizontalalignment='center')
        
        ax.legend(loc='lower left', fontsize='small')
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
                if key in p:
                    ans[key] = curve(p[key], p)
                else:
                    ans[key] = curve(x[key], p)
            ans['nabeta'] = curve(p['mass'], p)
            ans['nabetagamma'] = curve(p['mass'] + 1274.5, p)
            # ans['nabetagamma'] = curve(p['mass'], p) + curve(1274.5, p) - p['intercept']
            # ans['coco'] = curve(X['co117'], p) + curve(X['co133'], p) - p['intercept']
            return ans
        
        print('__________{:s} {} test fit__________'.format(label, part_label))
        test_result = {}
        delta_sigmas = []
        for test_x in X:
            test_p0 = {test_x: X[test_x]}
            test_p0.update(p0)
            test_fit = lsqfit.nonlinear_fit(data=(X, Y), fcn=fcn, p0=test_p0, debug=True)
            test_result[test_x] = test_fit.p[test_x]
            delta_sigma = (test_fit.pmean[test_x] - X[test_x]) / test_fit.psdev[test_x]
            delta_sigmas.append(delta_sigma)
            print('Test {:6s}: nominal {:6.1f}, fitted {:11s}, delta/sigma = {:.1f}, mass = {}'.format(test_x, X[test_x], test_fit.p[test_x], delta_sigma, test_fit.p['mass']))
        factor = np.sqrt(np.mean(np.array(delta_sigmas) ** 2))
        print('Rms of delta/sigma = {:.1f}\n'.format(factor))
    
        print('__________{:s} {} calibration and mass fit__________'.format(label, part_label))
        fit = lsqfit.nonlinear_fit(data=(X, Y), fcn=fcn, p0=p0, debug=True)
        test = change_error(fit.p['mass'], factor, corr=False)
        input_var['{}_{}_test'.format(label, part_label)] = test
        mass['{}_{}'.format(label, part_label)] = fit.p['mass'] + test
        
        # chi2_dof = fit.chi2 / fit.dof
        # if chi2_dof > 1:
        #     chi2 = gvar.gvar(0, fit.p['mass'].sdev * np.sqrt(fit.chi2 / fit.dof - 1))
        #     mass['{}_{}'.format(label, part_label)] += chi2
        #     input_var['{}_{}_chi2'.format(label, part_label)] = chi2
    
        print(fit.format(maxline=True))
            
        # plot
        keys = list(X.keys())
        X_plot = [X[key] for key in keys]
        Y_plot = [Y[key] for key in keys]
        if nparts == 1:
            cal_label = 'PMT %d' % channel
        else:
            cal_label = '{}_{}'.format(label, part_label)
        rt = axcal.errorbar(gvar.mean(X_plot), gvar.mean(Y_plot), xerr=gvar.sdev(X_plot), yerr=gvar.sdev(Y_plot), fmt='.', label=cal_label)
        color = rt[0].get_color()
        X_plot += [fit.p['mass'], fit.p['mass'] + 1274.5]
        Y_plot += [Y['nabeta'], Y['nabetagamma']]
        axcal.errorbar(fit.pmean['mass'], gvar.mean(Y['nabeta']), xerr=fit.psdev['mass'], yerr=gvar.sdev(Y['nabeta']), fmt='x', color=color)
        axcal.errorbar(fit.pmean['mass'] + 1274.5, gvar.mean(Y['nabetagamma']), xerr=fit.psdev['mass'], yerr=gvar.sdev(Y['nabetagamma']), fmt='x', color=color)
        xspace = np.linspace(np.min(gvar.mean(X_plot)), np.max(gvar.mean(X_plot)), 500)
        axcal.plot(xspace, curve(xspace, fit.pmean), '-', color=color)
        
coord = dict(
    nabeta=[500, 400],
    cs=[750, 200],
    co117=[1500, 400],
    nabetagamma=[1750, 800],
    coco=[2350, 650]
)
for key in coord:
    axcal.text(*coord[key], peak_labels[key], ha='center')
    
print(gvar.tabulate(mass))
print(gvar.fmt_errorbudget(mass, input_var))

axcal.legend(loc=0)
axcal.set_xlabel('valore nominale / fittato [keV]')
axcal.set_ylabel('media del picco [canale ADC]')
axcal.grid(linestyle=':')

fig.show()
figcal.show()
