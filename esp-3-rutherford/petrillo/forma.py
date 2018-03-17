import numpy as np
import matplotlib.pyplot as plt
import uncertainties as un
from uncertainties import unumpy as unp

theta, count, time = np.loadtxt('../dati/0316-forma.txt', unpack=True)
L = un.ufloat(4, 0.1)
D = un.ufloat(4, 0.1)

# vedi sul logbook <<angolo forma>> per questi calcoli
theta = unp.radians(unp.uarray(theta, 1))
alpha = np.sign(unp.nominal_values(theta)) * unp.arccos((L*unp.cos(theta) + D) / unp.sqrt(L**2 + 2*L*D*unp.cos(theta) + D**2))
alpha = unp.degrees(alpha)
count = unp.uarray(count, np.sqrt(count))
time = unp.uarray(time, 0.5) * 1e-3
rate = count / time

fig = plt.figure('forma')
fig.clf()
fig.set_tight_layout(True)

ax = fig.add_subplot(111)
ax.errorbar(unp.nominal_values(alpha), unp.nominal_values(rate), xerr=unp.std_devs(alpha), yerr=unp.std_devs(rate), fmt=',k')
ax.grid(linestyle=':')
ax.set_xlabel('angolo [°]')
ax.set_ylabel('rate [s$^{-1}$]')

fig.show()
