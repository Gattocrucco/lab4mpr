import numpy as np
from matplotlib import pyplot as plt
from matplotlib import colors
from matplotlib import patches
import lab4
import fit_peak
import gvar
import lsqfit
import copy
import cross

file_c2 = '../DAQ/0525_rtot_c2.txt'
file_ch1 = '../DAQ/0525_rtot_ch1.txt'
file_ch2 = '../DAQ/0525_rtot_ch2.txt'

# prepare figures
fig_c2 = plt.figure('Rtot25-c2')
fig_c2.clf()
fig_c2.set_tight_layout(True)
ax_c2, ax_diff = fig_c2.subplots(1, 2, sharex=True, sharey=True)

fig_ch = plt.figure('Rtot25-ch')
fig_ch.clf()
fig_ch.set_tight_layout(True)
ax_ch1 = fig_ch.add_subplot(121)
ax_ch2 = fig_ch.add_subplot(122)

# load data
ch1_c2, ch2_c2 = lab4.loadtxt(file_c2, unpack=True, usecols=(0, 5))
ch1, = lab4.loadtxt(file_ch1, unpack=True, usecols=(0,))
ch2, = lab4.loadtxt(file_ch2, unpack=True, usecols=(5,))

# plot
bins = np.arange(1150 // 8) * 8
H, _, _, im = ax_c2.hist2d(ch1_c2, ch2_c2, bins=bins, norm=colors.LogNorm(), cmap='jet')
fig_c2.colorbar(im, ax=ax_c2)

H1, _ = np.histogram(ch1, bins=bins)
H2, _ = np.histogram(ch2, bins=bins)

ax_ch1.errorbar((bins[1:] + bins[:-1]) / 2, H1, yerr=np.sqrt(H1), fmt=',')
ax_ch2.errorbar((bins[1:] + bins[:-1]) / 2, H2, yerr=np.sqrt(H2), fmt=',')

# ax_ch1.set_yscale('log')
# ax_ch2.set_yscale('log')

ax_ch1.set_xlabel('energia PMT 1 [canale ADC]')
ax_ch1.set_ylabel('conteggio')
ax_ch2.set_xlabel('energia PMT 2 [canale ADC]')

############ fit ############

scaler = dict(
    c2=[8166, 1734781],
    ch1=[98472, 1248541],
    ch2=[121955, 148400]
)
norm = {}
counts = {}

##### fit 2d

cuts = dict(
    gammabeta=[(940, 1040), (410, 460)],
    betagamma=[(450, 540), (810, 890)]
)

bkgs = dict(
    gammabeta=None,
    betagamma=None
)

rate_corr = scaler['c2'][0] / (scaler['c2'][1] / 1000) / np.sum(H)

hist = gvar.gvar(H, np.sqrt(H))

for key in cuts.keys():
    print('_____________{}_____________'.format(key))
    ax_c2.fill(
        [cuts[key][0][0], cuts[key][0][1], cuts[key][0][1], cuts[key][0][0]],
        [cuts[key][1][0], cuts[key][1][0], cuts[key][1][1], cuts[key][1][1]],
        facecolor='none',
        edgecolor='black'
    )
    norm[key] = np.sum((cuts[key][0][0] <= ch1_c2) & (cuts[key][0][1] >= ch1_c2) & (cuts[key][1][0] <= ch2_c2) & (cuts[key][1][1] >= ch2_c2))
    counts[key] = norm[key]
    norm[key] = gvar.gvar(norm[key], np.sqrt(norm[key])) * rate_corr

##### fit 1d

cuts = {
    ('beta', 1): [450, 550],
    ('gamma', 1): [900, 1050],
    ('beta', 2): [450, 550],
    ('gamma', 2): [850, 1000]
}

for key in cuts:
    print('_____________{}{}_____________'.format(key[0], key[1]))
    h = [H1, H2][key[1] - 1]
    hist = gvar.gvar(h, np.sqrt(h))
    cut = fit_peak.cut(bins, cuts[key]) & (h >= 5)
    kw = dict(cut=cut, bkg='exp', print_info=1, ax=[ax_ch1, ax_ch2][key[1] - 1], plot_kw=dict(scaley=False))
    outputs, inputs = fit_peak.fit_peak(bins, hist, npeaks=1, **kw)
    norm[key] = outputs['peak1_norm']
    rate_corr = scaler['ch' + str(key[1])][0] / (scaler['ch' + str(key[1])][1] / 1000) / np.sum(h)
    norm[key] *= 1 / (bins[1] - bins[0]) * rate_corr
    counts[key] = outputs['peak1_norm'] / (bins[1] - bins[0])

##### global fit

p0 = dict(
    Rp_beta1=norm['beta', 1] / 2,
    Rp_beta2=norm['beta', 2] / 2,
    p_gamma1=norm['gammabeta'] / norm['beta', 2],
    p_gamma2=norm['betagamma'] / norm['beta', 1],
    Rtot=norm['gamma', 1] * norm['beta', 2] / norm['gammabeta']
)

ratios = {
    'p_beta1_tot/p_beta1': cross.total(0.511) / cross.photoel(0.511) * gvar.gvar(1, 0.2), # calcolare meglio
    'p_gamma1_tot/p_gamma1': cross.total(1.275) / cross.photoel(1.275) * gvar.gvar(1, 0.2), # calcolare meglio
    'p_beta2_tot/p_beta2': cross.total(0.511) / cross.photoel(0.511) * gvar.gvar(1, 0.2), # calcolare meglio
    'p_gamma2_tot/p_gamma2': cross.total(1.275) / cross.photoel(1.275) * gvar.gvar(1, 0.2), # calcolare meglio
}

Y = {}
Y.update(norm)
Y.update(ratios)

p0.update(dict(
    Rp_beta1_tot  = Y['p_beta1_tot/p_beta1'  ] * p0['Rp_beta1' ],
    p_gamma1_tot = Y['p_gamma1_tot/p_gamma1'] * p0['p_gamma1'],
    Rp_beta2_tot  = Y['p_beta2_tot/p_beta2'  ] * p0['Rp_beta2' ],
    p_gamma2_tot = Y['p_gamma2_tot/p_gamma2'] * p0['p_gamma2']
))

def fcn(p):
    ans = {}
    ans['p_beta1_tot/p_beta1'] = p['Rp_beta1_tot'] / p['Rp_beta1']
    ans['p_gamma1_tot/p_gamma1'] = p['p_gamma1_tot'] / p['p_gamma1']
    ans['p_beta2_tot/p_beta2'] = p['Rp_beta2_tot'] / p['Rp_beta2']
    ans['p_gamma2_tot/p_gamma2'] = p['p_gamma2_tot'] / p['p_gamma2']
    ans['betagamma'] = 2 * p['Rp_beta1'] * p['p_gamma2']
    ans['gammabeta'] = 2 * p['Rp_beta2'] * p['p_gamma1']
    ans['beta', 1] = 2 * p['Rp_beta1'] * (1 - p['p_gamma1_tot'])
    ans['gamma', 1] = (p['Rtot'] - 2 * p['Rp_beta1_tot']) * p['p_gamma1']
    ans['beta', 2] = 2 * p['Rp_beta2'] * (1 - p['p_gamma2_tot'])
    ans['gamma', 2] = (p['Rtot'] - 2 * p['Rp_beta2_tot']) * p['p_gamma2']
    return ans

fit = lsqfit.nonlinear_fit(data=Y, fcn=fcn, p0=gvar.mean(p0), debug=True)
print(fit.format(maxline=True))

# print(gvar.fmt_partialsdev(fit.p, Y, percent=True))
print(gvar.tabulate(fit.p))

fig_ch.show()
fig_c2.show()
