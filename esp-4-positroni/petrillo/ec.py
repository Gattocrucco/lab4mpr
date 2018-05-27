import numpy as np
from matplotlib import pyplot as plt
from matplotlib import gridspec
from matplotlib import colors
import lab4
import fit_peak
import gvar
import lsqfit
import copy
import cross

file_c2 = '../DAQ/0511_ec2_c2.txt'
file_ch1 = '../DAQ/0511_ec2_ch1.txt'
file_ch2 = '../DAQ/0511_ec2_ch2.txt'

# prepare figures
fig = plt.figure('ec')
fig.clf()
fig.set_tight_layout(True)
G = gridspec.GridSpec(1, 7)
ax_c2 = fig.add_subplot(G[:,0:3])
ax_ch1 = fig.add_subplot(G[:,3:5])
ax_ch2 = fig.add_subplot(G[:,5:7], sharex=ax_c2, sharey=ax_ch1)

fig_diff = plt.figure('ec-diff')
fig_diff.clf()
ax_diff = fig_diff.add_subplot(111)

# load data
ch1_c2, ch2_c2 = lab4.loadtxt(file_c2, unpack=True, usecols=(0, 1))
ch1, = lab4.loadtxt(file_ch1, unpack=True, usecols=(0,))
ch2, = lab4.loadtxt(file_ch2, unpack=True, usecols=(1,))

# plot
bins = np.arange(0, 1150, 8)
H, _, _, im = ax_c2.hist2d(ch1_c2, ch2_c2, bins=bins, norm=colors.LogNorm(), cmap='jet')
fig.colorbar(im, ax=ax_c2)

ax_c2.set_xlabel('energia PMT 1 [canale ADC]')
ax_c2.set_ylabel('energia PMT 2 [canale ADC]')

H1, _, _ = ax_ch1.hist(ch1, bins=bins, histtype='step', log=True)
H2, _, _ = ax_ch2.hist(ch2, bins=bins, histtype='step', log=True)

ax_ch1.set_xlabel('energia PMT 1 [canale ADC]')
ax_ch1.set_ylabel('conteggio / 8 canali')
ax_ch2.set_xlabel('energia PMT 2 [canale ADC]')

ax_c2.legend(title='trigger coincidenze')
ax_ch1.legend(title='trigger PMT 1')
ax_ch2.legend(title='trigger PMT 2')

############ fit ############

scaler = dict(
    c2=[22453919, 1973765],
    ch1=[2384113, 71537],
    ch2=[2611310, 80400]
)
norm = {}
count = {}

##### fit 2d

cuts = dict(
    betabeta=[(260, 320), (200, 270)],
    gammabeta=[(700, 800), (200, 270)],
    gammabetabeta=[(1000, 1080), (200, 270)],
    betagamma=[(260, 320), (610, 700)],
    betagammabeta=[(260, 320), (860, 1000)]
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
    outputs, inputs = fit_peak.fit_peak_2d(bins, bins, hist, cut=cut, bkg=bkgs[key], print_info=1, ax_2d=ax_c2, ax_2d_diff=None if key == 'betagammabeta' else ax_diff, plot_cut=False, corr=key == 'betabeta')
    norm[key] = outputs['norm'] / (bins[1] - bins[0]) ** 2 * rate_corr
    count[key] = outputs['norm'] / (bins[1] - bins[0]) ** 2

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
    kw = dict(cut=cut, bkg='exp', print_info=1, ax=[ax_ch1, ax_ch2][key[1] - 1], plot_kw=dict(scaley=False, color='black', label=None))
    if key[0] == 'beta':
        outputs, inputs = fit_peak.fit_peak(bins, hist, npeaks=2, **kw)
        norm[key] = outputs['peak1_norm'] + outputs['peak2_norm']
    else:
        outputs, inputs = fit_peak.fit_peak(bins, hist, npeaks=1, **kw)
        norm[key] = outputs['peak1_norm']
    rate_corr = scaler['ch' + str(key[1])][0] / (scaler['ch' + str(key[1])][1] / 1000) / np.sum(h)
    count[key] = norm[key] / (bins[1] - bins[0])
    norm[key] *= 1 / (bins[1] - bins[0]) * rate_corr

##### global fit

radius = gvar.gvar(2.54, 0.01)
distance = gvar.gvar(4.5, 0.1) + gvar.gvar(0.3, 0.1)
distance_1 = distance + radius * 2/3
distance_2 = distance + radius * 4/3
cos_theta = distance_1 / np.sqrt(distance_1 ** 2 + radius ** 2)
acc_1 = 1/2 * (1 - cos_theta)
cos_theta = distance_2 / np.sqrt(distance_2 ** 2 + radius ** 2)
acc_2 = 1/2 * (1 - cos_theta)

acc1 = gvar.gvar(gvar.mean(acc_1+acc_2) / 2, gvar.mean(acc_1 - acc_2))
acc2 = copy.copy(acc1)
acc12 = copy.copy(acc1)

# ratios = {
#     'p_beta1_tot/p_beta1': cross.total(0.511) / cross.photoel(0.511) * gvar.gvar(1, 0.2), # calcolare meglio
#     'p_gamma1_tot/p_gamma1': cross.total(1.275) / cross.photoel(1.275) * gvar.gvar(1, 0.2), # calcolare meglio
#     'p_beta2_tot/p_beta2': cross.total(0.511) / cross.photoel(0.511) * gvar.gvar(1, 0.2), # calcolare meglio
#     'p_gamma2_tot/p_gamma2': cross.total(1.275) / cross.photoel(1.275) * gvar.gvar(1, 0.2), # calcolare meglio
# }
#
# ratios['p_beta12_tot/p_beta12'] = ratios['p_beta1_tot/p_beta1'] * ratios['p_beta2_tot/p_beta2']
#
# Y = {}
# Y.update(norm)
# Y.update(ratios)
#
# p0 = dict(
#     Rp_beta12=norm['betabeta'],
#     Rp_beta1=norm['beta', 1] / 2,
#     Rp_beta2=norm['beta', 2] / 2,
#     p_gamma1=norm['gammabeta'] / norm['beta', 2],
#     p_gamma2=norm['betagamma'] / norm['beta', 1],
#     R_tot=norm['gamma', 1] * norm['beta', 2] / norm['gammabeta']
# )
#
# p0.update(dict(
#     Rp_beta1_tot =  Y['p_beta1_tot/p_beta1'  ] * p0['Rp_beta1' ],
#     p_gamma1_tot =  Y['p_gamma1_tot/p_gamma1'] * p0['p_gamma1' ],
#     Rp_beta2_tot =  Y['p_beta2_tot/p_beta2'  ] * p0['Rp_beta2' ],
#     p_gamma2_tot =  Y['p_gamma2_tot/p_gamma2'] * p0['p_gamma2' ],
#     Rp_beta12_tot = Y['p_beta12_tot/p_beta12'] * p0['Rp_beta12']
# ))
#
# def fcn(p):
#     ans = {}
#
#     R_tot = p['R_tot']
#     Rp_beta1 = p['Rp_beta1']
#     Rp_beta2 = p['Rp_beta2']
#     p_gamma1 = p['p_gamma1']
#     p_gamma2 = p['p_gamma2']
#     Rp_beta12 = p['Rp_beta12']
#     Rp_beta1_tot = p['Rp_beta1_tot']
#     Rp_beta2_tot = p['Rp_beta2_tot']
#     p_gamma1_tot = p['p_gamma1_tot']
#     p_gamma2_tot = p['p_gamma2_tot']
#     Rp_beta12_tot = p['Rp_beta12_tot']
#
#     ans['p_beta1_tot/p_beta1'] = Rp_beta1_tot / Rp_beta1
#     ans['p_gamma1_tot/p_gamma1'] = p_gamma1_tot / p_gamma1
#     ans['p_beta2_tot/p_beta2'] = Rp_beta2_tot / Rp_beta2
#     ans['p_gamma2_tot/p_gamma2'] = p_gamma2_tot / p_gamma2
#     ans['p_beta12_tot/p_beta12'] = Rp_beta12_tot / Rp_beta12
#
#     ans['beta', 1] = 2 * Rp_beta1 * (1 - p_gamma1_tot)
#     ans['gamma', 1] = R_tot * p_gamma1 - 2 * Rp_beta1_tot * p_gamma1
#     ans['gammabeta'] = 2 * Rp_beta2 * p_gamma1 - Rp_beta12_tot * p_gamma1
#     ans['gammabetabeta'] = Rp_beta12 * p_gamma1
#
#     ans['beta', 2] = 2 * Rp_beta2 * (1 - p_gamma2_tot)
#     ans['gamma', 2] = R_tot * p_gamma2 - 2 * Rp_beta2_tot * p_gamma2
#     ans['betagamma'] = 2 * Rp_beta1 * p_gamma2 - Rp_beta12_tot * p_gamma2
#     ans['betagammabeta'] = Rp_beta12 * p_gamma2
#
#     ans['betabeta'] = Rp_beta12 * (1 - p_gamma1_tot - p_gamma2_tot)
#
#     return ans

ratios = {
    # 'p_beta_tot/p_beta': cross.total(0.511) / cross.photoel(0.511) * gvar.gvar(1, 0.2), # calcolare meglio
    # 'p_gamma_tot/p_gamma': cross.total(1.275) / cross.photoel(1.275) * gvar.gvar(1, 0.2), # calcolare meglio
    'p_beta_tot/p_beta': 1 / gvar.gvar(0.50, 0.02) * gvar.gvar(1, 0.1), # dal Knoll
    'p_gamma_tot/p_gamma': 1 / gvar.gvar(0.22, 0.02) * gvar.gvar(1, 0.1) # dal Knoll
}

ratios['p_beta12_tot/p_beta12'] = 2 * ratios['p_beta_tot/p_beta'] - 1

Y = {}
Y.update(norm)
Y.update(ratios)

p0 = dict(
    Rp_beta12=norm['betabeta'],
    Rp_beta=norm['beta', 1] / 2,
    p_gamma=norm['gammabeta'] / norm['beta', 2],
    R_tot=norm['gamma', 1] * norm['beta', 2] / norm['gammabeta']
)

p0.update(dict(
    Rp_beta_tot =  ratios['p_beta_tot/p_beta'  ] * p0['Rp_beta' ],
    p_gamma_tot =  ratios['p_gamma_tot/p_gamma'] * p0['p_gamma' ],
    Rp_beta12_tot = ratios['p_beta12_tot/p_beta12'] * p0['Rp_beta12']
))

def fcn(p):
    ans = {}

    R_tot = p['R_tot']
    Rp_beta = p['Rp_beta']
    p_gamma = p['p_gamma']
    Rp_beta12 = p['Rp_beta12']
    Rp_beta_tot = p['Rp_beta_tot']
    p_gamma_tot = p['p_gamma_tot']
    Rp_beta12_tot = p['Rp_beta12_tot']
    
    ans['p_beta_tot/p_beta'] = Rp_beta_tot / Rp_beta
    ans['p_gamma_tot/p_gamma'] = p_gamma_tot / p_gamma
    ans['p_beta12_tot/p_beta12'] = Rp_beta12_tot / Rp_beta12

    ans['beta', 1] = 2 * Rp_beta * (1 - p_gamma_tot)
    ans['gamma', 1] = R_tot * p_gamma - 2 * Rp_beta_tot * p_gamma
    ans['gammabeta'] = 2 * Rp_beta * p_gamma - Rp_beta12_tot * p_gamma
    ans['gammabetabeta'] = Rp_beta12 * p_gamma
    
    ans['beta', 2] = 2 * Rp_beta * (1 - p_gamma_tot)
    ans['gamma', 2] = R_tot * p_gamma - 2 * Rp_beta_tot * p_gamma
    ans['betagamma'] = 2 * Rp_beta * p_gamma - Rp_beta12_tot * p_gamma
    ans['betagammabeta'] = Rp_beta12 * p_gamma
    
    ans['betabeta'] = Rp_beta12 * (1 - 2 * p_gamma_tot)
    
    return ans

fit = lsqfit.nonlinear_fit(data=Y, fcn=fcn, p0=gvar.mean(p0), debug=True)
print(fit.format(maxline=True))

R = acc12 / (acc1 * acc2) * fit.p['Rp_beta'] ** 2 / fit.p['Rp_beta12']

rat = R / fit.p['R_tot']

input_var = {}
input_var.update(data=list(norm.values()), ratios=list(ratios.values()), accs=[acc1, acc2, acc12])
output_var = dict(R=R, Rtot=fit.p['R_tot'], R_Rtot=rat)

print(gvar.tabulate(output_var))
print(gvar.fmt_errorbudget(output_var, input_var, percent=False))

fig.show()
fig_diff.show()
