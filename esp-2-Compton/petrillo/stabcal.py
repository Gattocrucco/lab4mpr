import numpy as np
from matplotlib import pyplot as plt
import histo
from collections import OrderedDict

# script che analizza la misura di controllo della stabilità della calibrazione della notte 22-23 febbraio

slices = 24

all_samples = np.load('../dati/log-23feb-notte.npy')

cuts = np.array(np.round(np.arange(slices + 1) / slices * len(all_samples)), dtype='int32')

samples_dict = OrderedDict()
for i in range(len(cuts) - 1):
    samples = all_samples[cuts[i]:cuts[i+1]]
    samples_dict['slice%d' % (i+1,)] = samples

histo.histo(samples_dict, logscale=False, figname='23feb-notte', cut=10)
