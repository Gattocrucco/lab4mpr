import numpy as np
from matplotlib import pyplot as plt
from matplotlib import colors
import lab4
import fit_peak
import gvar
import lsqfit
import copy

file_c2 = '../DAQ/0524_rtot_prova1.txt'
file_ch1 = '../DAQ/0524_rtot_prova1_ch1.txt'
file_ch2 = '../DAQ/0524_rtot_prova1_ch2.txt'

# prepare figures
fig_c2 = plt.figure('Rtot24-c2')
fig_c2.clf()
fig_c2.set_tight_layout(True)
ax_c2, ax_diff = fig_c2.subplots(1, 2, sharex=True, sharey=True)

fig_ch = plt.figure('Rtot24-ch')
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

input_var = {}

scaler = dict(
    c2=[22884, 3122739],
    ch1=[121096, 1607176],
    ch2=[153030, 180619]
)
norm = {}

##### fit 2d

cuts = dict(
    betabeta=[(475, 560), (425, 500)],
    gammabeta=[(970, 1055), (430, 500)]
)

bkgs = dict(
    betabeta='expcross',
    gammabeta=None
)

rate_corr = scaler['c2'][0] / (scaler['c2'][1] / 1000) / np.sum(H)

hist = gvar.gvar(H, np.sqrt(H))

for key in cuts.keys():
    print('_____________{}_____________'.format(key))
    cut = fit_peak.cut_2d(bins, bins, *cuts[key]) & (H >= 5)
    outputs, inputs = fit_peak.fit_peak_2d(bins, bins, hist, cut=cut, bkg=bkgs[key], print_info=1, ax_2d=ax_c2, ax_2d_diff=ax_diff, plot_cut=True, corr=key == 'betabeta')
    norm[key] = outputs['norm'] / (bins[1] - bins[0]) ** 2 * rate_corr
    input_var[key] = inputs['data']

##### fit 1d

cuts = {
    ('beta', 1): [480, 540],
    ('beta', 2): [475, 550],
    ('gamma', 1): [950, 1050],
    ('gamma', 2): [875, 1000]
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
    input_var[key] = inputs['data']

##### global fit

p0 = dict(
    Rp_beta1=10,
    Rp_beta1_tot=10,
    p_gamma1=0.01,
    p_gamma1_tot=0.01,
    Rp_beta2=100,
    Rp_beta2_tot=100,
    p_gamma2=0.1,
    p_gamma2_tot=0.1,
    Rp_beta12=50,
    Rp_beta12_tot=50,
    Rtot=1000
)

Y = {
    'p_beta12_tot/p_beta12': (95/15) ** 2 * gvar.gvar(1, 0.3),
    'p_beta1_tot/p_beta1': 95/15 * gvar.gvar(1, 0.2), # calcolare meglio
    'p_gamma1_tot/p_gamma1': 51/2 * gvar.gvar(1, 0.2), # calcolare meglio
    'p_beta2_tot/p_beta2': 95/15 * gvar.gvar(1, 0.2), # calcolare meglio
    'p_gamma2_tot/p_gamma2': 51/2 * gvar.gvar(1, 0.2), # calcolare meglio
}
Y.update(norm)

def fcn(p):
    ans = {}
    ans['p_beta12_tot/p_beta12'] = p['Rp_beta12_tot'] / p['Rp_beta12']
    ans['p_beta1_tot/p_beta1'] = p['Rp_beta1_tot'] / p['Rp_beta1']
    ans['p_gamma1_tot/p_gamma1'] = p['p_gamma1_tot'] / p['p_gamma1']
    ans['p_beta2_tot/p_beta2'] = p['Rp_beta2_tot'] / p['Rp_beta2']
    ans['p_gamma2_tot/p_gamma2'] = p['p_gamma2_tot'] / p['p_gamma2']
    ans['gammabeta'] = 2 * p['Rp_beta2'] * p['p_gamma1'] - p['Rp_beta12_tot'] * p['p_gamma1']
    ans['betabeta'] = p['Rp_beta12'] * (1 - p['p_gamma1_tot'] - p['p_gamma2_tot'])
    ans['beta', 1] = 2 * p['Rp_beta1'] * (1 - p['p_gamma1_tot'])
    ans['gamma', 1] = (p['Rtot'] - 2 * p['Rp_beta1_tot']) * p['p_gamma1']
    ans['beta', 2] = 2 * p['Rp_beta2'] * (1 - p['p_gamma2_tot'])
    ans['gamma', 2] = (p['Rtot'] - 2 * p['Rp_beta2_tot']) * p['p_gamma2']
    return ans

fit = lsqfit.nonlinear_fit(data=Y, fcn=fcn, p0=p0, debug=True)
print(fit.format(maxline=True))

print(gvar.fmt_partialsdev(fit.p, input_var, percent=True))
print(gvar.tabulate(fit.p))

fig_ch.show()
fig_c2.show()
