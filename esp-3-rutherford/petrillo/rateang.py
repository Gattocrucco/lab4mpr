import numpy as np
import matplotlib.pyplot as plt
from uncertainties import unumpy as unp
import lab4
import lab

ang, count, clock = np.loadtxt('../dati/0320-angoli.txt', unpack=True)

count = unp.uarray(count, np.sqrt(count))
time = unp.uarray(clock, 0.5) * 1e-3
rate = count / time
ang = unp.uarray(ang, 1)

f = lambda ang, amp, center: amp / np.sin(np.radians(ang - center) / 2) ** 4
p0 = [0.0001, 1]
out = lab.fit_curve(f, ang, rate, p0=p0, print_info=1)

fig = plt.figure('rateang')
fig.clf()
fig.set_tight_layout(True)

ax1 = fig.add_subplot(111)
fang = np.linspace(np.min(unp.nominal_values(ang)), np.max(unp.nominal_values(ang)), 500)
lab4.errorbar(ang, rate, ax=ax1, fmt=',k')
ax1.plot(fang, f(fang, *p0), '-r', scaley=False)
ax1.set_xlabel('angolo [°]')
ax1.set_ylabel('rate [s$^{-1}$]')
ax1.grid(linestyle=':')

fig.show()
