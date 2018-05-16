import numpy as np
import lsqfit
import gvar
import fit_peak
from matplotlib import pyplot as plt
import lab4

fig = plt.figure('mass')
fig.clf()
fig.set_tight_layout(True)
axs = fig.subplots(3, 3)

scaler = dict(
    ch1=dict(
        # count, clock
        na=[139078, 549517],
        co=[144058, 230065],
        cs=[153727, 118537],
        cocs=[285774, 208510],
        noise=[14174, 156138]
    ),
    ch2=dict(
        na=[53689, 234084],
        co=[96191, 152209],
        cs=[111208, 82846],
        cocs=[300772, 215507],
        noise=[8695, 141072]
    ),
    ch3=dict(
        na=[84684, 370713],
        co=[83020, 152873],
        cs=[120790, 103970],
        cocs=[257807, 129724],
        noise=[5911, 82183]
    )
)

cut = dict(
    ch1=dict(
        # left, right
        nabeta=[275, 325],
        nagamma=[675, 775],
        co=[625, 800],
        cs=[350, 425]
    ),
    ch2=dict(
        nabeta=[300, 350],
        nagamma=[675, 775],
        co=[625, 800],
        cs=[375, 450]
    ),
    ch3=dict(
        nabeta=[325, 400],
        nagamma=[675, 775],
        co=[650, 775],
        cs=[425, 480]
    )
)

minlength = 1150
rebin = 8
bins = np.arange(minlength + 1)[::rebin]
x = (bins[:-1] + bins[1:]) / 2

input_var = {}

for channel in [1, 2, 3]:
    label = 'ch{:d}'.format(channel)
    
    # noise
    filename = '../DAQ/0515_noise_{:s}{:s}.txt'.format(label, '_conbarattolo' if channel == 1 else '')
    samples, = lab4.loadtxt(filename, usecols=(0,), unpack=True, dtype='uint16')
    hist = lab4.rebin(np.bincount(samples, minlength=minlength), rebin)
    cut5_noise = hist >= 5
    hist = gvar.gvar(hist, np.sqrt(hist))
    rate_noise = hist * (scaler[label]['noise'][0] / len(samples)) / (scaler[label]['noise'][1] / 1000)
    input_var[label + '_noise'] = hist
    
    peaks = {}
    for source_idx in range(3):
        source = ['na', 'co', 'cs'][source_idx]
        ax = axs[channel - 1][source_idx]
        
        # load data
        filename = '../DAQ/0515_{:s}_{:s}.txt'.format(source, label)
        samples, = lab4.loadtxt(filename, usecols=(0,), unpack=True, dtype='uint16')
        hist = lab4.rebin(np.bincount(samples, minlength=minlength), rebin)
        cut5 = (hist >= 5)
        hist = gvar.gvar(hist, np.sqrt(hist))
        rate = hist * (scaler[label][source][0] / len(samples)) / (scaler[label][source][1] / 1000)
        input_var['{:s}_{:s}'.format(label, source)] = hist
        corr_rate = rate - rate_noise
        
        # plot
        ax.errorbar(x, gvar.mean(corr_rate), yerr=gvar.sdev(corr_rate), fmt=',', label=label + ' ' + source)
        # lab4.bar(bins, gvar.mean(rate_noise), ax=ax, label='fondo')
        ax.set_yscale('log')
        if source == 'na':
            ax.set_ylabel('rate [s$^{-1}$]')
        if channel == 3:
            ax.set_xlabel('canale ADC')

        # fit
        kw = dict(ax=ax, plot_kw=dict(scaley=False), print_info=1)
        if kw['print_info']:
            print('_________{:s}_{:s}_________'.format(label, source))
        if source == 'na':
            cut_margins = cut[label]['nabeta']
            cut_bool = (cut_margins[0] <= x) & (x <= cut_margins[1]) & cut5
            outputs, inputs = fit_peak.fit_peak(bins, corr_rate, cut=cut_bool, bkg='exp', **kw)
            peaks['nabeta'] = outputs['peak1_mean']
            
            cut_margins = cut[label]['nagamma']
            cut_bool = (cut_margins[0] <= x) & (x <= cut_margins[1]) & cut5
            outputs, inputs = fit_peak.fit_peak(bins, corr_rate, cut=cut_bool, bkg='exp', **kw)
            peaks['nagamma'] = outputs['peak1_mean']
        elif source == 'co':
            cut_margins = cut[label]['co']
            cut_bool = (cut_margins[0] <= x) & (x <= cut_margins[1]) & cut5
            mean1 = cut_margins[0] + (cut_margins[1] - cut_margins[0]) * 1/3
            mean2 = cut_margins[0] + (cut_margins[1] - cut_margins[0]) * 2/3
            manual_p0 = {}#dict(peak1_mean=mean1, peak2_mean=mean2, peak1_sigma=10, peak2_sigma=10)
            outputs, inputs = fit_peak.fit_peak(bins, corr_rate, npeaks=2, cut=cut_bool, bkg='exp', manual_p0=manual_p0, **kw)
            peak12 = [outputs['peak1_mean'], outputs['peak2_mean']]
            idx = np.argsort(gvar.mean(peak12))
            peaks['co117'] = peak12[idx[0]]
            peaks['co133'] = peak12[idx[1]]
        elif source == 'cs':
            cut_margins = cut[label]['cs']
            cut_bool = (cut_margins[0] <= x) & (x <= cut_margins[1]) & cut5
            outputs, inputs = fit_peak.fit_peak(bins, corr_rate, cut=cut_bool, bkg='exp', **kw)
            peaks['cs'] = outputs['peak1_mean']
        
        ax.legend(loc='upper right', fontsize='small')

fig.show()
