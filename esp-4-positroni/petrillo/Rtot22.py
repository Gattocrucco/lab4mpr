import numpy as np
from matplotlib import pyplot as plt
from matplotlib import colors
import lab4
import fit_peak
import gvar
import lsqfit
import copy

file_c2 = '../DAQ/0522_prova_rtot_attaccati.txt'
file_ch1 = '../DAQ/0522_prova_rtot_attaccati_ch1.txt'
file_ch2 = '../DAQ/0522_prova_rtot_attaccati_ch2.txt'

# prepare figures
fig_c2 = plt.figure('Rtot-c2')
fig_c2.clf()
fig_c2.set_tight_layout(True)
ax_c2, ax_diff = fig_c2.subplots(1, 2, sharex=True, sharey=True)

fig_ch = plt.figure('Rtot-ch')
fig_ch.clf()
fig_ch.set_tight_layout(True)
ax_ch1 = fig_ch.add_subplot(121)
ax_ch2 = fig_ch.add_subplot(122)

# load data
ch1_c2, ch2_c2 = lab4.loadtxt(file_c2, unpack=True, usecols=(0, 11))
ch1, = lab4.loadtxt(file_ch1, unpack=True, usecols=(0,))
ch2, = lab4.loadtxt(file_ch2, unpack=True, usecols=(11,))

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
    c2=[32010, 516352],
    ch1=[64697, 140308],
    ch2=[102327, 232830]
)
norm = {}

##### fit 2d

cuts = dict(
    betagamma=[(440, 510), (810, 890)],
    gammabeta=[(900, 980), (390, 450)]
)

bkgs = dict(
    betagamma=None,
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
    ('beta', 1): [460, 530],
    ('beta', 2): [460, 550],
    ('gamma', 1): [900, 1000],
    ('gamma', 2): [880, 990]
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

results = dict(
    p_gamma1 = norm['gammabeta'] / norm['beta', 2],
    p_gamma2 = norm['betagamma'] / norm['beta', 1],
    R_tot_1 = norm['gamma', 1] * norm['beta', 2] / norm['gammabeta'],
    R_tot_2 = norm['gamma', 2] * norm['beta', 1] / norm['betagamma']
)

print(gvar.fmt_partialsdev(results, input_var, percent=True))
print(gvar.tabulate(results))

fig_ch.show()
fig_c2.show()
