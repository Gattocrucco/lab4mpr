import numpy as np
import matplotlib.pyplot as plt
import uncertainties as un
from uncertainties import unumpy as unp

ang, count, time = np.loadtxt('../dati/0316-forma.txt', unpack=True)

ang = unp.uarray(ang, 1)
count = unp.uarray(count, np.sqrt(count))
time = unp.uarray(time, 0.5)
rate = count / time

fig = plt.figure('forma')
fig.clf()
fig.set_tight_layout(True)

ax = fig.add_subplot(111)
ax.errorbar(unp.nominal_values(ang), unp.nominal_values(rate), xerr=unp.std_devs(ang), yerr=unp.std_devs(rate), fmt='.k')
ax.grid(linestyle=':')
ax.set_xlabel('angolo [°]')
ax.set_ylabel('rate [s$^{-1}$]')

fig.show()
