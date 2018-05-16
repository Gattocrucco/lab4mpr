import numpy as np
import lsqfit
import gvar
import fit_peak
from matplotlib import pyplot as plt
import lab4

fig = plt.figure('mass')
fig.clf()
fig.set_tight_layout(True)
axs = fig.subplots(3, 3, sharex='col')

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
bins = np.arange(minlength + 1)

for channel in [1, 2, 3]:
    label = 'ch{:d}'.format(channel)
    
    # noise
    filename = '../DAQ/0515_noise_{:s}{:s}.txt'.format(label, '_conbarattolo' if channel == 1 else '')
    samples, = lab4.loadtxt(filename, usecols=(0,), unpack=True, dtype='uint16')
    hist = np.bincount(samples, minlength=minlength)
    rate_noise = hist * (scaler[label]['noise'][0] / len(samples)) / (scaler[label]['noise'][1] / 1000)
    rate_noise = gvar.gvar(rate_noise, np.sqrt(rate_noise))
    
    for source_idx in range(3):
        source = ['na', 'co', 'cs'][source_idx]
        ax = axs[channel - 1][source_idx]
        
        # load data
        filename = '../DAQ/0515_{:s}_{:s}.txt'.format(source, label)
        samples, = lab4.loadtxt(filename, usecols=(0,), unpack=True, dtype='uint16')
        hist = np.bincount(samples, minlength=minlength)
        rate = hist * (scaler[label][source][0] / len(samples)) / (scaler[label][source][1] / 1000)
        rate = gvar.gvar(rate, np.sqrt(rate))
        
        # # fit
        # if source == 'na':
        #     cut_margins = cut[label]['nabeta']
        #     outputs, inputs = fit_peak.fit_peak(bins, cut=cut_margins)
        #
        # plot
        lab4.bar(bins, gvar.mean(rate), ax=ax, label=label + ' ' + source)
        lab4.bar(bins, gvar.mean(rate_noise), ax=ax, label='fondo')
        ax.set_yscale('log')
        ax.legend(loc='upper right', fontsize='small')
        if source == 'na':
            ax.set_ylabel('rate [s$^{-1}$]')
        if channel == 3:
            ax.set_xlabel('canale ADC')
    
fig.show()
