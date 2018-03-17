import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
import uncertainties as un
from uncertainties import unumpy as unp
import lab4
from scipy import stats
from scipy import optimize

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

def credible_interval(samples, cl=0.68, ax=None):
    kde = stats.gaussian_kde(samples)
    pdf = kde(samples)
    idx = np.argsort(pdf)
    sorted_samples = samples[idx]
    interval_samples = sorted_samples[-int(np.round(cl * len(samples))):]
    left = np.min(interval_samples)
    right = np.max(interval_samples)
    act_cl = len(interval_samples) / len(samples)
    out = optimize.minimize_scalar(lambda x: -kde(x), bracket=(left, right))
    if not out.success:
        raise RuntimeError('can not find mode of pdf')
    if not (ax is None):
        ax.plot(samples, pdf, '.k', label='samples')
        l = ax.get_ylim()
        ax.plot(2 * [out.x[0]], l, '--k', scaley=False, label='mode')
        rect = patches.Rectangle(
            (left, l[0]),
            right - left,
            l[1] - l[0],
            facecolor='lightgray',
            edgecolor='none',
            zorder=-1,
            label='%.3f CR' % act_cl
        )
        ax.add_patch(rect)
        ax.legend(loc='best', fontsize='small')
        ax.grid(linestyle=':')
        ax.set_xlabel('value')
        ax.set_ylabel('pdf')
    return out.x[0], left, right, act_cl

fig = plt.figure('forma-spettro')
fig.clf()
fig.set_tight_layout(True)
ax = fig.add_subplot(111)

s = []
serr = [[], []]
cl = []
for angle in ang:
    filename = '../de0_data/0316ang{}.dat'.format('{}'.format(int(angle)).replace('-', '_'))
    print('processing {}...'.format(filename))
    t, ch1, ch2 = np.loadtxt(filename, unpack=True)
    out = credible_interval(ch1, cl=0.9, ax=None if angle != 0 else ax)
    s.append(out[0])
    serr[0].append(out[0] - out[1])
    serr[1].append(out[2] - out[0])
    cl.append(out[3])

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
