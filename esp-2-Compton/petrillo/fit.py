import numpy as np
import mc9
import empirical
import matplotlib.pyplot as plt
import histo

pa, sa = mc9.mc_cached(1.33, theta_0=40, N=100000, seed=0)
pb, sb = mc9.mc_cached(1.17, theta_0=40, N=100000, seed=1)

samples = np.concatenate([pa, sa, pb, sb])

counts = np.bincount(np.asarray(np.floor(samples), dtype='u2'), minlength=2**13)[:2**13]

fig = plt.figure('fit')
fig.clf()
ax = fig.add_subplot(111)

histo.bar_line(np.arange(2**13 + 1), counts, ax=ax)

fig.show()
