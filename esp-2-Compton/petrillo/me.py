import pickle
from matplotlib import pyplot as plt
from matplotlib import gridspec
import numpy as np
import uncertainties as un
from uncertainties import unumpy as unp
import calibration
import bias
import sympy as sp

with open('fit-result.pickle', 'rb') as load_file:
    centers_133, centers_117, centers_133_sim, centers_117_sim, theta_0s, calib_date, fixnorm, logcut = pickle.load(load_file)

def fun_energy(E_0, m_e, theta_0):
    return E_0 / (1 + E_0 / m_e * (1 - np.cos(np.radians(theta_0))))

def weighted_mean(y):
    inv_covy = np.linalg.inv(un.covariance_matrix(y))
    vara = 1 / np.sum(inv_covy)
    a = vara * np.sum(np.dot(inv_covy, y))
    return a

cntcal_133     = []
cntcal_117     = []
cntcal_133_sim = []
cntcal_117_sim = []
for i in range(len(calib_date)):
    cntcal_133    .append(calibration.energy_inverse_calibration(calib_date[i], unc=True)(centers_133[i])    )
    cntcal_117    .append(calibration.energy_inverse_calibration(calib_date[i], unc=True)(centers_117[i])    )
    cntcal_133_sim.append(calibration.energy_inverse_calibration(calib_date[i], unc=True)(centers_133_sim[i]))
    cntcal_117_sim.append(calibration.energy_inverse_calibration(calib_date[i], unc=True)(centers_117_sim[i]))

utheta_0s = np.array([un.ufloat(t, 0.1, tag='angle') for t in theta_0s]) - un.ufloat(-0.09, 0.04, tag='forma')
m_133 = np.array([(1 - un.umath.cos(un.umath.radians(utheta_0s[i]))) / (1 / cntcal_133[i] - 1 / 1.33) for i in range(len(utheta_0s))])
m_117 = np.array([(1 - un.umath.cos(un.umath.radians(utheta_0s[i]))) / (1 / cntcal_117[i] - 1 / 1.17) for i in range(len(utheta_0s))])
m_133_sim = np.array([(1 - un.umath.cos(un.umath.radians(utheta_0s[i]))) / (1 / cntcal_133_sim[i] - 1 / 1.33) for i in range(len(utheta_0s))])
m_117_sim = np.array([(1 - un.umath.cos(un.umath.radians(utheta_0s[i]))) / (1 / cntcal_117_sim[i] - 1 / 1.17) for i in range(len(utheta_0s))])

biases = np.array([bias.bias_double(1.33, 1.17, theta_0s[i], calib_date[i], fixnorm=fixnorm[i]) for i in range(len(theta_0s))])

m_133 -= biases[:,0]
m_117 -= biases[:,1]
m_133_sim -= biases[:,0]
m_117_sim -= biases[:,1]

fig = plt.figure('me')
fig.clf()
fig.set_tight_layout(True)
G = gridspec.GridSpec(1, 3)
ax = fig.add_subplot(G[:,:2])

s = 0.15
ax.errorbar(np.arange(len(theta_0s)) - 3*s/2, unp.nominal_values(m_133_sim), yerr=unp.std_devs(m_133_sim), fmt='^', label='1.33 fondo semplificato', color=[0.6]*3)
ax.errorbar(np.arange(len(theta_0s)) +   s/2, unp.nominal_values(m_117_sim), yerr=unp.std_devs(m_117_sim), fmt='v', label='1.17 fondo semplificato', color=[0.6]*3)
ax.errorbar(np.arange(len(theta_0s)) -   s/2, unp.nominal_values(m_133), yerr=unp.std_devs(m_133), fmt='^', label='1.33', color=[0]*3)
ax.errorbar(np.arange(len(theta_0s)) + 3*s/2, unp.nominal_values(m_117), yerr=unp.std_devs(m_117), fmt='v', label='1.17', color=[0]*3)
ax.set_xticks(np.arange(len(theta_0s)))
labels = []
for i in range(len(theta_0s)):
    label = '{}°'.format(theta_0s[i])
    if not (logcut[i] is None):
        c = 2**5 * 3**3 * 5**4
        s = sp.Rational(float(np.round(logcut[i][0] * c))) / c
        e = sp.Rational(float(np.round(logcut[i][1] * c))) / c
        label += '\n${}$-${}$'.format(sp.latex(s), sp.latex(e))
    labels.append(label)
ax.set_xticklabels(labels)
ax.legend(loc='best', fontsize='small')
ax.grid(linestyle=':')
ax.set_xlabel('presa dati: angolo, partizione')
ax.set_ylabel('massa dell’elettrone [MeV]')

ax = fig.add_subplot(G[:,-1])

fig.show()
