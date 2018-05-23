import numpy as np
from matplotlib import pyplot as plt
from matplotlib import gridspec
from matplotlib import colors
import lab4
import fit_peak
import gvar
import lsqfit
import copy

file_c2 = '../DAQ/0511_ec2_c2.txt'
file_ch1 = '../DAQ/0511_ec2_ch1.txt'
file_ch2 = '../DAQ/0511_ec2_ch2.txt'

# prepare figures
fig_c2 = plt.figure('ec-c2')
fig_c2.clf()
fig_c2.set_tight_layout(True)
G = gridspec.GridSpec(3, 3)
ax_c2 = fig_c2.add_subplot(G[:-1,1:])
ax_c2_ch1 = fig_c2.add_subplot(G[-1,1:], sharex=ax_c2)
ax_c2_ch2 = fig_c2.add_subplot(G[:-1,0], sharey=ax_c2)
ax_colorbar = fig_c2.add_subplot(G[-1,0])

fig_ch = plt.figure('ec-ch')
fig_ch.clf()
fig_ch.set_tight_layout(True)
ax_ch1 = fig_ch.add_subplot(121)
ax_ch2 = fig_ch.add_subplot(122)

fig_diff = plt.figure('ec-diff')
fig_diff.clf()
ax_diff = fig_diff.add_subplot(111)

# load data
ch1_c2, ch2_c2 = lab4.loadtxt(file_c2, unpack=True, usecols=(0, 1))
ch1, = lab4.loadtxt(file_ch1, unpack=True, usecols=(0,))
ch2, = lab4.loadtxt(file_ch2, unpack=True, usecols=(1,))

# plot
bins = np.arange(1150 // 8) * 8
H, _, _, im = ax_c2.hist2d(ch1_c2, ch2_c2, bins=bins, norm=colors.LogNorm(), cmap='jet')
fig_c2.colorbar(im, ax=ax_colorbar, fraction=0.5, aspect=2)
ax_colorbar.axis('off')
ax_c2_ch1.hist(ch1_c2, bins=bins, orientation='vertical', histtype='step', log=True)
ax_c2_ch2.hist(ch2_c2, bins=bins, orientation='horizontal', histtype='step', log=True)

ax_c2_ch1.set_xlabel('energia PMT 1 [canale ADC]')
ax_c2_ch2.set_ylabel('energia PMT 2 [canale ADC]')

H1, _, _ = ax_ch1.hist(ch1, bins=bins, histtype='step', log=True)
H2, _, _ = ax_ch2.hist(ch2, bins=bins, histtype='step', log=True)

ax_ch1.set_xlabel('energia PMT 1 [canale ADC]')
ax_ch2.set_xlabel('energia PMT 2 [canale ADC]')

############ fit ############

scaler = dict(
    c2=[22453919, 1973765],
    ch1=[2384113, 71537],
    ch2=[2611310, 80400]
)
norm = {}

##### fit 2d

cuts = dict(
    betabeta=[(260, 320), (200, 270)],
    gammabeta=[(700, 800), (200, 270)],
    gammabetabeta=[(1000, 1080), (200, 270)],
    betagamma=[(260, 320), (610, 700)],
    betagammabeta=[(260, 320), (900, 1000)]
)

bkgs = dict(
    betabeta='expcross',
    gammabeta='expx',
    gammabetabeta='expx',
    betagamma='expy',
    betagammabeta='expy'
)

rate_corr = scaler['c2'][0] / (scaler['c2'][1] / 1000) / np.sum(H)

# use density to keep confrontable 1d and 2d histograms
hist = gvar.gvar(H, np.sqrt(H))

for key in cuts.keys():
    print('_____________{}_____________'.format(key))
    cut = fit_peak.cut_2d(bins, bins, *cuts[key]) & (H >= 5)
    outputs, inputs = fit_peak.fit_peak_2d(bins, bins, hist, cut=cut, bkg=bkgs[key], print_info=1, ax_2d=ax_c2, ax_2d_diff=ax_diff, plot_cut=True, corr=key == 'betabeta')
    norm[key] = outputs['norm'] / (bins[1] - bins[0]) ** 2 * rate_corr

##### fit 1d

cuts = {
    ('beta', 1): [250, 370],
    ('gamma', 1): [690, 870],
    ('beta', 2): [200, 300],
    ('gamma', 2): [620, 750]
}

for key in cuts:
    print('_____________{}{}_____________'.format(key[0], key[1]))
    h = [H1, H2][key[1] - 1]
    hist = gvar.gvar(h, np.sqrt(h))
    cut = fit_peak.cut(bins, cuts[key]) & (h >= 5)
    kw = dict(cut=cut, bkg='exp', print_info=1, ax=[ax_ch1, ax_ch2][key[1] - 1], plot_kw=dict(scaley=False))
    if key[0] == 'beta':
        outputs, inputs = fit_peak.fit_peak(bins, hist, npeaks=2, **kw)
        norm[key] = outputs['peak1_norm'] + outputs['peak2_norm']
    else:
        outputs, inputs = fit_peak.fit_peak(bins, hist, npeaks=1, **kw)
        norm[key] = outputs['peak1_norm']
    rate_corr = scaler['ch' + str(key[1])][0] / (scaler['ch' + str(key[1])][1] / 1000) / np.sum(h)
    norm[key] *= 1 / (bins[1] - bins[0]) * rate_corr

##### global fit

# norm.pop('gammabeta')
# norm.pop('betagamma')
# norm.pop('betabeta')

p0_1 = {
    'Racc': 1e6,
    'R_tot': 11e6,
    'p_beta1': 0.1,
    'p_beta2': 0.1,
    'p_gamma1acc': 0.1,
    'p_gamma2acc': 0.1,
}

radius = gvar.gvar(2.54, 0.01)
distance = gvar.gvar(4.5, 0.1)
cos_theta = distance / np.sqrt(distance ** 2 + radius ** 2)
acc = 1/2 * (1 - cos_theta)

def fcn_1(p):
    ans = {}
    
    Racc = p['Racc']
    R_tot = p['R_tot']
    p_beta1 = p['p_beta1']
    p_beta2 = p['p_beta2']
    p_gamma1 = p['p_gamma1acc']
    p_gamma2 = p['p_gamma2acc']
    
    Rp_beta12 = Racc * p_beta1 * p_beta2
    Rp_beta1 = 2 * Racc * p_beta1
    Rp_beta2 = 2 * Racc * p_beta2
    
    ans['beta', 1] = Rp_beta1 * (1 - p_gamma1)
    ans['gamma', 1] = R_tot * p_gamma1 - Rp_beta1 * p_gamma1
    ans['gammabeta'] = Rp_beta2 * p_gamma1 - Rp_beta12 * p_gamma1
    ans['gammabetabeta'] = Rp_beta12 * p_gamma1
    
    ans['beta', 2] = Rp_beta2 * (1 - p_gamma2)
    ans['gamma', 2] = R_tot * p_gamma2 - Rp_beta2 * p_gamma2
    ans['betagamma'] = Rp_beta1 * p_gamma2 - Rp_beta12 * p_gamma2
    ans['betagammabeta'] = Rp_beta12 * p_gamma2
    
    ans['betabeta'] = Rp_beta12 * (1 - p_gamma1 - p_gamma2)
    
    
    for key in copy.copy(ans):
        if not (key in norm):
            ans.pop(key)
    
    return ans

p0_2 = {
    'R_tot': 11e6,
    'Rp_beta1': 10e6 * 0.1,
    'Rp_beta2': 10e6 * 0.1,
    'p_gamma1': 0.1,
    'p_gamma2': 0.1,
    'Rp_beta12': 10e6 * 0.1
}

def fcn_2(p):
    ans = {}

    R_tot = p['R_tot']
    Rp_beta1 = p['Rp_beta1'] * 2
    Rp_beta2 = p['Rp_beta2'] * 2
    p_gamma1 = p['p_gamma1']
    p_gamma2 = p['p_gamma2']
    Rp_beta12 = p['Rp_beta12']
    
    ans['beta', 1] = Rp_beta1 * (1 - p_gamma1)
    ans['gamma', 1] = R_tot * p_gamma1 - Rp_beta1 * p_gamma1
    ans['gammabeta'] = Rp_beta2 * p_gamma1 - Rp_beta12 * p_gamma1
    ans['gammabetabeta'] = Rp_beta12 * p_gamma1
    
    ans['beta', 2] = Rp_beta2 * (1 - p_gamma2)
    ans['gamma', 2] = R_tot * p_gamma2 - Rp_beta2 * p_gamma2
    ans['betagamma'] = Rp_beta1 * p_gamma2 - Rp_beta12 * p_gamma2
    ans['betagammabeta'] = Rp_beta12 * p_gamma2
    
    ans['betabeta'] = Rp_beta12 * (1 - p_gamma1 - p_gamma2)
    
    return ans

fit = lsqfit.nonlinear_fit(data=norm, fcn=fcn_2, p0=p0_2, debug=True)
print(fit.format(maxline=True))

fig_ch.show()
fig_c2.show()
fig_diff.show()
