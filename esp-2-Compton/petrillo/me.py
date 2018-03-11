import pickle
from matplotlib import pyplot as plt
from matplotlib import gridspec
from matplotlib import patches
import numpy as np
import uncertainties as un
from uncertainties import unumpy as unp
import calibration
import bias
import sympy as sp
import copy
import collections
import lab

##### LOAD FILE #####

with open('fit-result.pickle', 'rb') as load_file:
    centers_133, centers_117, centers_133_sim, centers_117_sim, theta_0s, calib_date, fixnorm, logcut = pickle.load(load_file)
    theta_0s = np.array(theta_0s)
    logcut = np.array(logcut)

##### FUNCTIONS #####

def fun_energy(E_0, m_e, theta_0):
    return E_0 / (1 + E_0 / m_e * (1 - np.cos(np.radians(theta_0))))

def errorsummary(x):
    comps = x.error_components()
    
    tags = set(map(lambda v: v.tag, comps.keys()))
    var = dict(zip(tags, [0] * len(tags)))
    
    for (v, sd) in comps.items():
        var[v.tag] += sd ** 2
    
    tags = list(tags)
    sds = np.sqrt(np.array([var[tag] for tag in tags]))
    idx = np.argsort(sds)[::-1]
    d = collections.OrderedDict()
    for i in idx:
        d[tags[i]] = sds[i]
    
    return d

def weighted_mean(y):
    """
    y is array of ufloats
    """
    inv_covy = np.linalg.inv(un.covariance_matrix(y))
    vara = 1 / np.sum(inv_covy)
    a = vara * np.sum(np.dot(inv_covy, y))
    assert np.allclose(vara, a.s ** 2)
    
    res = unp.nominal_values(y) - a.n
    Q = float(res.reshape(1,-1) @ inv_covy @ res.reshape(-1,1))
    
    return a, Q

##### COMPUTE ALL MASSES #####

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

##### COMPUTE FINAL MEASUREMENT #####

# 15°, start
m_133_15 = m_133[theta_0s == 15][0]
m_117_15 = m_117[theta_0s == 15][0]

m_133_15_p = m_133[theta_0s == 15][1]
m_117_15_p = m_117[theta_0s == 15][1]

stab_133_15 = abs(m_133_15_p.n - m_133_15.n) / m_133_15.n
stab_117_15 = abs(m_117_15_p.n - m_117_15.n) / m_117_15.n
stab_15 = un.ufloat(1, max(stab_133_15, stab_117_15), tag='stability')
stab_15_time = (logcut[theta_0s == 15][0][1] - logcut[theta_0s == 15][0][0]) * 22
m_133_15 *= stab_15
m_117_15 *= stab_15

bias_133_15 = biases[:,0][theta_0s == 15][0]
bias_117_15 = biases[:,1][theta_0s == 15][0]

# 15°, end
m_133_15e = m_133[theta_0s == 15][4]
m_117_15e = m_117[theta_0s == 15][4]

m_133_15e_p = m_133[theta_0s == 15][5]
m_117_15e_p = m_117[theta_0s == 15][5]

stab_133_15e = abs(m_133_15e_p.n - m_133_15e.n) / m_133_15e.n
stab_117_15e = abs(m_117_15e_p.n - m_117_15e.n) / m_117_15e.n
stab_15e = un.ufloat(1, max(stab_133_15e, stab_117_15e), tag='stability')
m_133_15e *= stab_15e
m_117_15e *= stab_15e

bias_133_15e = biases[:,0][theta_0s == 15][4]
bias_117_15e = biases[:,1][theta_0s == 15][4]

# 61.75°
stab_61_time = 40
stab_61 = un.ufloat(1, stab_15.s * (1 - np.cos(np.radians(15))) / (1 - np.cos(np.radians(61.75))) * np.sqrt(stab_61_time / stab_15_time), tag='stability')

m_133_61 = m_133_sim[theta_0s == 61.75][0]
m_133_61 *= stab_61
m_117_61 = m_117_sim[theta_0s == 61.75][0]
m_117_61 *= stab_61

bias_133_61 = biases[:,0][theta_0s == 61.75][0]
bias_117_61 = biases[:,1][theta_0s == 61.75][0]

# 45°
stab_45_time = 20
stab_45 = un.ufloat(1, stab_15.s * (1 - np.cos(np.radians(15))) / (1 - np.cos(np.radians(45))) * np.sqrt(stab_45_time / stab_15_time), tag='stability')

m_133_45 = m_133[theta_0s == 45][0]
m_133_45 *= stab_45
m_117_45 = m_117[theta_0s == 45][0]
m_117_45 *= stab_45

bias_133_45 = biases[:,0][theta_0s == 45][0]
bias_117_45 = biases[:,1][theta_0s == 45][0]

# weighted mean
masses_133 = np.array([m_133_15, m_133_15e, m_133_61, m_133_45])
masses_117 = np.array([m_117_15, m_117_15e, m_117_61, m_117_45])
masses = np.concatenate([masses_133, masses_117])
biases = np.array([bias_133_15, bias_133_15e, bias_133_61, bias_133_45, bias_117_15, bias_117_15e, bias_117_61, bias_117_45])

me,     Q = weighted_mean(masses)
me_133, _ = weighted_mean(masses_133)
me_117, _ = weighted_mean(masses_117)

angles = np.array([15, 15, 61.75, 45])

# impose 133 and 117 compatibility
d = me_133 - me_117
s = np.sqrt((d.n ** 2 - d.s ** 2) / 2)
mec_133 = me_133 * un.ufloat(1, s / me_133.n, tag='comp')
mec_117 = me_117 * un.ufloat(1, s / me_117.n, tag='comp')
dc = mec_133 - mec_117
assert np.allclose(abs(dc.n), dc.s)
mec, Qc = weighted_mean([mec_133, mec_117])
assert np.allclose(Qc, 1)

##### SUMMARY #####

print('massa 1.33: {:P}'.format(me_133))
print('massa 1.17: {:P}'.format(me_117))
print('    differenza: {:P} ({:.1f} σ)'.format(d, abs(d.n) / d.s))
print('massa vera: 0.511')
print('massa     : {:P}'.format(me))
dv = me - 0.511
print('    differenza: {:P} ({:.1f} σ)'.format(dv, abs(dv.n) / dv.s))
print("componenti dell'incertezza:")
for k, s in errorsummary(me).items():
    print('    {:11}: {:.3g}'.format(str(k), s))
print('massa comp: {:P}'.format(mec))
dv = mec - 0.511
print('    differenza: {:P} ({:.1f} σ)'.format(dv, abs(dv.n) / dv.s))
print("componenti dell'incertezza:")
for k, s in errorsummary(mec).items():
    print('    {:11}: {:.3g}'.format(str(k), s))
print('chi2/ndof = {:.1f} / {} = {:.1f}'.format(Q, len(masses) - 1, Q / (len(masses) - 1)))
print('matrice di covarianza:')
print(lab.format_par_cov(unp.nominal_values(masses), un.covariance_matrix(masses)))

##### PLOT #####

fig = plt.figure('me', figsize=[8.35, 3.41])
fig.clf()
fig.set_tight_layout(True)
G = gridspec.GridSpec(1, 3)
ax = fig.add_subplot(G[:,:2])

s = 0.15
ax.errorbar(np.arange(len(theta_0s)) - 3*s/2, unp.nominal_values(m_133_sim), yerr=unp.std_devs(m_133_sim), fmt='^', label='1.33 fondo semplif.', color=[0.6]*3)
ax.errorbar(np.arange(len(theta_0s)) +   s/2, unp.nominal_values(m_117_sim), yerr=unp.std_devs(m_117_sim), fmt='v', label='1.17 fondo semplif.', color=[0.6]*3)
ax.errorbar(np.arange(len(theta_0s)) -   s/2, unp.nominal_values(m_133), yerr=unp.std_devs(m_133), fmt='^', label='1.33', color=[0]*3)
ax.errorbar(np.arange(len(theta_0s)) + 3*s/2, unp.nominal_values(m_117), yerr=unp.std_devs(m_117), fmt='v', label='1.17', color=[0]*3)
ax.set_xticks(np.arange(len(theta_0s)))
labels = []
for i in range(len(theta_0s)):
    label = '{:g}°'.format(theta_0s[i])
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
ax.errorbar(np.arange(len(angles)) - s/2, unp.nominal_values(masses_133), yerr=unp.std_devs(masses_133), fmt='^k', label='1.33')
ax.errorbar(np.arange(len(angles)) + s/2, unp.nominal_values(masses_117), yerr=unp.std_devs(masses_117), fmt='vk', label='1.17')
lim = ax.get_xlim()
rect = patches.Rectangle(
    (lim[0], me.n - me.s),
    lim[1] - lim[0],
    2 * me.s,
    facecolor='lightgray',
    edgecolor='none',
    zorder=-1,
    label='media $\\pm 1 \\sigma$'
)
ax.add_patch(rect)
ax.plot(lim, [0.511] * 2, '--k', scalex=False, label='valore noto')
ax.legend(loc='best', fontsize='small')
ax.grid(linestyle=':')
ax.set_xticks(np.arange(len(angles)))
ax.set_xticklabels(['{:g}°'.format(a) for a in angles])
ax.set_xlabel('presa dati: angolo')

fig.show()
