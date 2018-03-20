import numpy as np
import matplotlib.pyplot as plt
import uncertainties as un
from uncertainties import unumpy as unp
import lab4

###### RATE vs. ANGLE ######

ang, count, time = np.loadtxt('../dati/0316-forma.txt', unpack=True)
diameter = un.ufloat(16.3, 0.1)
overhang = un.ufloat(5.3, 0.1)
source = un.ufloat(3.1, 0.1)
L = diameter / 2 - overhang # cm
D = source # cm

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

fig = plt.figure('forma-spettro')
fig.clf()
fig.set_tight_layout(True)
ax = fig.add_subplot(111)

s, cl, lengths = np.empty((3, len(ang)))
serr = np.empty((2, len(ang)))
for i in range(len(ang)):
    angle = ang[i]
    filename = '../de0_data/0316ang{}.dat'.format('{}'.format(int(angle)).replace('-', '_'))
    if angle == 80:
        filename = '../de0_data/0319ang80new.dat'
    print('processing {}...'.format(filename))
    t, ch1, ch2 = np.loadtxt(filename, unpack=True)
    out = lab4.credible_interval(ch1, cl=0.9, ax=None if angle != 0 else ax)
    s[i] = out[0]
    serr[0, i] = out[0] - out[1]
    serr[1, i] = out[2] - out[0]
    cl[i] = out[3]
    lengths[i] = len(ch1)
    if abs(len(ch1) - count[i].n) / count[i].n > 0.001:
        print('warning: angle {:.2f}: ADC and scaler counts differ more than 0.1 %'.format(angle))

###### PLOT ######

fig.show()

fig = plt.figure('forma')
fig.clf()
fig.set_tight_layout(True)

ax = fig.add_subplot(211)
lab4.errorbar(alpha, rate, ax=ax, fmt=',k')
ax.grid(linestyle=':')
ax.set_ylabel('rate [s$^{-1}$ cm$^{2}$]')

ax = fig.add_subplot(212)
lab4.errorbar(alpha, s, yerr=serr, ax=ax, fmt=',k')
ax.grid(linestyle=':')
ax.set_xlabel('angolo [°]')
ax.set_ylabel('moda (90 % CR) [digit]')

fig.show()
