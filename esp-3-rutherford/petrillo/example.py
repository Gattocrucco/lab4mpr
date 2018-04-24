import fit
import matplotlib.pyplot as plt
import lab4
import numpy as np

fig = plt.figure('example')
fig.clf()
fig.set_tight_layout(True)

ax = fig.add_subplot(111)

edges, counts, dummy1, dummy2 = fit.load_spectrum('0319notte.dat', None)

counts = counts / np.diff(edges)
lab4.bar(edges, counts, ax=ax)

ax.set_xlabel('canale ADC [digit]')
ax.set_ylabel('conteggio [digit$^{-1}$]')
ax.grid(linestyle=':')
ax.set_yscale('symlog', linthreshy=np.min(counts[counts > 0]))

fig.show()
