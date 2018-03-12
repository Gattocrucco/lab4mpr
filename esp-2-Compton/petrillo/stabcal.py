import numpy as np
from matplotlib import pyplot as plt
import histo
from collections import OrderedDict

# script che analizza la misura di controllo della stabilità della calibrazione della notte 22-23 febbraio

slices = 5

all_samples = np.load('../dati/log-26feb-stab.npy')

cuts = np.array(np.round(np.arange(slices + 1) / slices * len(all_samples)), dtype='int32')

samples_dict = OrderedDict()
for i in range(len(cuts) - 1):
    samples = all_samples[cuts[i]:cuts[i+1]]
    samples_dict['slice%d' % (i+1,)] = samples

fig = plt.figure('stability', figsize=[6.88, 2.93])
fig.clf()
fig.set_tight_layout(True)
ax = fig.add_subplot(111)

histo.histo(samples_dict, cut=8, linewidth=1.5, ax=ax)
ax.grid(linestyle=':')
ax.legend(['%d h-%d h' % (h, h + 5) for h in range(0, 25, 5)], fontsize='small')

fig.show()
