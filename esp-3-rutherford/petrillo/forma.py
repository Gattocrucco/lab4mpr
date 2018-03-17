import numpy as np
import matplotlib.pyplot as plt
import uncertainties as un
from uncertainties import unumpy as unp
import lab4

###### RATE vs. ANGLE ######

ang, count, time = np.loadtxt('../dati/0316-forma.txt', unpack=True)
L = un.ufloat(4, 0.1) # cm
D = un.ufloat(4, 0.1) # cm

# vedi sul logbook <<angolo forma>> per questi calcoli
theta = unp.radians(unp.uarray(ang, 1))
X = unp.sqrt(L**2 + 2*L*D*unp.cos(theta) + D**2)
alpha = np.sign(unp.nominal_values(theta)) * unp.arccos((L*unp.cos(theta) + D) / X)

# theta = 0 va propagato a parte
theta_0 = theta[unp.nominal_values(theta) == 0]
alpha_0 = L/(L + D) * theta_0
alpha[unp.nominal_values(theta) == 0] = alpha_0

alpha = unp.degrees(alpha)
count = unp.uarray(count, np.sqrt(count))
time = unp.uarray(time, 0.5) * 1e-3
rate = count / time * X**2

###### SPECTRUM vs. ANGLE ######

s = []
for angle in ang:
    t, ch1, ch2 = np.loadtxt('../de0_data/0316ang{}.dat'.format('{}'.format(int(angle)).replace('-', '_')), unpack=True)
    s.append(un.ufloat(np.mean(ch1), np.std(ch1, ddof=1)))
s = np.array(s)

###### PLOT ######

fig = plt.figure('forma')
fig.clf()
fig.set_tight_layout(True)

ax = fig.add_subplot(211)
lab4.errorbar(alpha, rate, ax=ax, fmt=',k')
ax.grid(linestyle=':')
ax.set_ylabel('rate [s$^{-1}$ cm$^{2}$]')

ax = fig.add_subplot(212)
lab4.errorbar(alpha, s, ax=ax, fmt=',k')
ax.grid(linestyle=':')
ax.set_xlabel('angolo [°]')
ax.set_ylabel('media ± sdev [digit]')

fig.show()
