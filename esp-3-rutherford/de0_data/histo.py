# -*- coding: utf-8 -*-
from __future__ import division, print_function
import numpy as np
import matplotlib.pyplot as plt
import sys

# legge i files da linea di comando
filenames = sys.argv[1:]

fig = plt.figure('histo')
fig.clf()
fig.set_tight_layout(True)

def unroll_time(t):
    if len(t) == 1: 
        return t
    tmax = 6553.5
    # preso da max(t)
    # bisogna sommare 65535 e non 65536 perché min(t) == 0.1
    diff = np.diff(t)
    cycles = np.concatenate([[0], np.cumsum(diff < 0)])
    return t + tmax * cycles

def find_noise(t):
    # t deve essere già "srotolato"
    # restituisce un array di indici che sono rumori

    # controlla che il rate sia abbastanza basso,
    # altrimenti è normale che ci siano eventi con lo stesso timestamp
    rate = len(t) / (np.max(t) - np.min(t))
    if rate > 1/60: # più di uno al minuto
        return np.zeros(len(t), dtype=bool)

    # cerca gruppi di eventi consecutivi con lo stesso timestamp
    dt_zero = np.diff(t) == 0
    noise = np.concatenate([dt_zero, [False]]) | np.concatenate([[False], dt_zero])
    return noise

if len(filenames) > 1:
    datasets = []
    for filename in filenames:
        print('loading %s...' % (filename,))
        t, ch1, ch2 = np.loadtxt(filename, unpack=True).reshape(3, -1)
        datasets.append(ch1)
    ax = fig.add_subplot(111)
    nbinspow = min(int(np.ceil(np.log2(np.sqrt(max([len(ds) for ds in datasets]))))), 12)
    edges = np.arange(2 ** 12 + 1)[::2 ** (12 - nbinspow)] - 0.5
    ax.hist(datasets, bins=edges, density=True, histtype='step', label=filenames)
    ax.legend(loc='upper right', fontsize='small')
    ax.set_xlabel('canale ADC')
    ax.set_ylabel('densita')

elif len(filenames) == 1:
    filename = filenames[0]
    print('loading %s...' % (filename,))
    rolled_t, ch1, ch2 = np.loadtxt(filename, unpack=True).reshape(3, -1)
    t = unroll_time(rolled_t)
    noise = find_noise(t)
    noise_ch2 = ch2 != 0
    
    ax1 = fig.add_subplot(211)
    hist_data = ch1[~(noise | noise_ch2)]
    nbinspow = min(int(np.ceil(np.log2(np.sqrt(len(hist_data))))), 12)
    edges = np.arange(2 ** 12 + 1)[::2 ** (12 - nbinspow)] - 0.5
    ax1.hist(hist_data, bins=edges, histtype='step', label=filename)
    ax1.legend(loc='upper right', fontsize='small')
    ax1.set_ylabel('conteggio [no rumori]')
    ax1.set_xlabel('canale ADC')
    
    ax2 = fig.add_subplot(212)
    ax2.plot(t, ch1, '.', markersize=2)
    ax2.plot(t[noise], ch1[noise], 'rx', label='timestamp')
    ax2.plot(t[noise_ch2], ch1[noise_ch2], 'k+', label='ch2 > 0')
    ax2.set_xlabel('tempo [s]')
    ax2.set_ylabel('canale ADC')
    ax2.legend(fontsize='small', loc='best')

else:
    print('no filenames specified.')

fig.show()

