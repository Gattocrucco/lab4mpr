import histo
import numpy as np
from matplotlib import pyplot as plt
import scanf

files = ['../dati/histo-16feb-ang0.dat', '../dati/histo-16feb-ang90.dat']

fig = plt.figure('plotg', figsize=[6.88, 2.93])
fig.clf()
fig.set_tight_layout(True)

for i in range(len(files)):
    counts = np.loadtxt(files[i], unpack=True, dtype='u2')
    edges = np.arange(2 ** 13 + 1)
    
    angle, = scanf.scanf('%*sang%d.dat', files[i])
    
    rebin = 32
    counts = histo.partial_sum(counts, rebin)
    edges = edges[::rebin]
    
    ax = fig.add_subplot(1, 2, i + 1)
    histo.bar_line(edges, counts, ax=ax, color='black')
    if i == 0:
        ax.set_ylabel('conteggio [(%d$\\cdot$digit)$^{-1}$]' % (rebin,))
    ax.set_xlabel('canale ADC [digit]')
    ax.set_title('spettro a {}°'.format(angle))
    ax.grid(linestyle=':')

fig.show()
