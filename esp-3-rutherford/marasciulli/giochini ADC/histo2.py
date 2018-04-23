# -*- coding: utf-8 -*-
from __future__ import division, print_function
import numpy as np
import matplotlib.pyplot as plt
import sys
import lab4

# legge i files da linea di comando
filenames = '0420prova_all50.dat'

fig = plt.figure()
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


filename = filenames
print('loading %s...' % (filename,))
rolled_t, ch1, ch2 = np.loadtxt(filename, unpack=True).reshape(3, -1)

t = unroll_time(rolled_t)
noise = find_noise(t)
ax1 = fig.add_subplot(111)
#nbinspow = min(int(np.ceil(np.log2(np.sqrt(len(ch1))))), 12)
#edges = np.arange(2 ** 12 + 1)[::2 ** (12 - nbinspow)] - 0.5

# taglietti
'''
# vanno bene per l'oro
pezzo1=(2389.2,2407.8) 
pezzo2=(2407.9,2426)
pezzo3=(2426.1,2474.3)
pezzo4=(2474.4,2525.9)
pezzo5=(2526,2553.4)

guad50=pezzo1+pezzo5
guad100=pezzo2+pezzo4
guad200=pezzo3
'''

# vanno bene per l'alluminio
sett1=(1976.7,1994.6)
sett2=(1994.7,2037.2)
sett3=(2037.3,2063.6)
sett4=(2063.7,2099.8)
sett5=(2099.9,2117.8)

guad25=sett4
guad50=sett1+sett3+sett5
guad100=sett2


settore=eval("guad25")

cond=logical_and(t<settore[1] ,t>settore[0])
ch11=ch1[cond]
t1=t[cond]

# grafici

ax1.set_ylabel('conteggio')
ax1.set_xlabel('canale ADC')

k=np.arange(64)*2
ax1.hist(ch11+0.5, bins=k*32+1,histtype='step',label="guadagno 25")


settore=eval("guad50")

cond=((t>settore[0]) & (t<settore[1])) | ((t>settore[2]) & (t<settore[3])) | ((t>settore[4]) & (t<settore[5]))
ch11=ch1[cond]
t1=t[cond]

ax1.hist(ch11+0.5, bins=k*32+1,histtype='step',label="guadagno 50")


settore=eval("guad100")

cond= (t<settore[1]) & (t>settore[0])
ch11=ch1[cond]
t1=t[cond]

ax1.hist(ch11+0.5, bins=k*32+1,histtype='step',label="guadagno 100")

ax1.legend(loc=0,fontsize='x-small')
fig.show()