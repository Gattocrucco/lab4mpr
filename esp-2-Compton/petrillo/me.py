import pickle
from matplotlib import pyplot as plt
import numpy as np
import uncertainties as un
from uncertainties import unumpy as unp
import calibration
import bias

with open('fit-result.pickle', 'rb') as load_file:
    centers_133, centers_117, centers_133_sim, centers_117_sim, theta_0s, calib_date, fixnorm = pickle.load(load_file)

def fun_energy(E_0, m_e, theta_0):
    return E_0 / (1 + E_0 / m_e * (1 - np.cos(np.radians(theta_0))))

for i in range(len(calib_date)):
    centers_133[i] = calibration.energy_inverse_calibration(calib_date[i], unc=True)(centers_133[i])
    centers_117[i] = calibration.energy_inverse_calibration(calib_date[i], unc=True)(centers_117[i])
    centers_133_sim[i] = calibration.energy_inverse_calibration(calib_date[i], unc=True)(centers_133_sim[i])
    centers_117_sim[i] = calibration.energy_inverse_calibration(calib_date[i], unc=True)(centers_117_sim[i])

utheta_0s = np.array([un.ufloat(t, 0.1, tag='angle') for t in theta_0s]) - un.ufloat(-0.09, 0.04, tag='forma')
m_133 = np.array([(1 - un.umath.cos(un.umath.radians(utheta_0s[i]))) / (1 / centers_133[i] - 1 / 1.33) for i in range(len(utheta_0s))])
m_117 = np.array([(1 - un.umath.cos(un.umath.radians(utheta_0s[i]))) / (1 / centers_117[i] - 1 / 1.17) for i in range(len(utheta_0s))])
m_133_sim = np.array([(1 - un.umath.cos(un.umath.radians(utheta_0s[i]))) / (1 / centers_133_sim[i] - 1 / 1.33) for i in range(len(utheta_0s))])
m_117_sim = np.array([(1 - un.umath.cos(un.umath.radians(utheta_0s[i]))) / (1 / centers_117_sim[i] - 1 / 1.17) for i in range(len(utheta_0s))])

biases = np.array([bias.bias_double(1.33, 1.17, theta_0s[i], calib_date[i], fixnorm=fixnorm[i]) for i in range(len(theta_0s))])

m_133 -= biases[:,0]
m_117 -= biases[:,1]
m_133_sim -= biases[:,0]
m_117_sim -= biases[:,1]

fig = plt.figure('me')
fig.clf()
ax = fig.add_subplot(111)

ax.errorbar(np.arange(len(theta_0s)) - 0.05, unp.nominal_values(m_133), yerr=unp.std_devs(m_133), fmt='.', label='1.33')
ax.errorbar(np.arange(len(theta_0s)) + 0.05, unp.nominal_values(m_117), yerr=unp.std_devs(m_117), fmt='.', label='1.17')
ax.errorbar(np.arange(len(theta_0s)) - 0.1, unp.nominal_values(m_133_sim), yerr=unp.std_devs(m_133_sim), fmt='.', label='1.33 simp.')
ax.errorbar(np.arange(len(theta_0s)) + 0.1, unp.nominal_values(m_117_sim), yerr=unp.std_devs(m_117_sim), fmt='.', label='1.17 simp.')
ax.set_xticks(np.arange(len(theta_0s)))
ax.set_xticklabels(theta_0s)
ax.legend(loc=1)

fig.show()
