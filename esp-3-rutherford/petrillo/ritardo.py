import numpy as np
import matplotlib.pyplot as plt

thresh, ant, unc_ant = np.loadtxt('../dati/0319-ritardo.txt', unpack=True)

fig = plt.figure('ritardo')
fig.clf()
fig.set_tight_layout(True)
ax = fig.add_subplot(111)

ax.errorbar(thresh, ant, yerr=unc_ant, fmt='.k')

ax.set_xlabel('soglia discriminatore')
ax.set_ylabel('ritardo segnale discriminato [ns]')
ax.grid(linestyle=':')

fig.show()
