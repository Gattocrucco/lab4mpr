from __future__ import division, print_function
import numpy as np
import matplotlib.pyplot as plt
import uncertainties as un
from uncertainties import unumpy as unp
import lab4
import sys

direct = '../de0_data/'

AU = 'oro.dat'
ALL = 'all.dat'
FONDO = direct+'0417buco10_08.dat'
CAL = direct+'0416buco18_35.dat'

filenames = [AU, ALL, CAL]
title = ['oro', 'alluminio', 'Am241']

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
    ax.hist(datasets, bins=edges, density=True, histtype='step', label=title)
    ax.legend(loc='upper right', fontsize='small')
    ax.set_xlabel('canale ADC')
    ax.set_ylabel('densita')

elif len(filenames) == 1:
    filename = filenames[0]
    print('loading %s...' % (filename,))
    rolled_t, ch1, ch2 = np.loadtxt(filename, unpack=True).reshape(3, -1)
    t = unroll_time(rolled_t)
    noise = find_noise(t)
    ax1 = fig.add_subplot(211)
    nbinspow = min(int(np.ceil(np.log2(np.sqrt(len(ch1))))), 12)
    edges = np.arange(2 ** 12 + 1)[::2 ** (12 - nbinspow)] - 0.5
    ax1.hist(ch1, bins=edges, histtype='step', label=title)
    ax1.legend(loc='upper right', fontsize='small')
    ax1.set_ylabel('conteggio')
    ax1.set_xlabel('canale ADC')
    ax2 = fig.add_subplot(212)
    ax2.plot(t, ch1, '.', markersize=2)
    ax2.plot(t[noise], ch1[noise], 'rx')
    ax2.set_xlabel('tempo')
    ax2.set_ylabel('canale ADC')

else:
    print('no filenames specified.')
rate = len(t) / (np.max(t) - np.min(t))
fig.show()

for name in filenames:
    t, ch1, ch2 = np.loadtxt(name, unpack=True)
    #plt.figure(name)
    #plt.hist(ch1)
    out = lab4.credible_interval(ch1, cl=0.68)
    print(name+":"+str(out[0])+"+"+str(out[2]-out[0])+"-"+str(out[1]-out[0]))
    