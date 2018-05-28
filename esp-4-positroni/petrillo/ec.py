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
import lab

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

# things for the latex tables
def fmt(u):
    return '\\num{' + lab.util_format(gvar.mean(u), gvar.sdev(u), pm=None, comexp=True, dot=False) + '}'
nothing = '-'

##### fit 2d

peak_2d_fit_table = []

cuts = dict(
    betabeta=[(260, 320), (200, 270)],
    gammabeta=[(700, 800), (200, 270)],
    gammabetabeta=[(1000, 1080), (200, 270)],
    betagamma=[(260, 320), (610, 700)],
    betagammabeta=[(260, 320), (860, 1000)]
)

labels = {key: '${}$'.format(key.replace('gammabetabeta', 'gammabeta,beta').replace('betagammabeta', 'beta,gammabeta').replace('beta', '\\beta').replace('gamma', '\\gamma')) for key in cuts}

bkgs = dict(
    betabeta='expcross',
    gammabeta='expx',
    gammabetabeta='expx',
    betagamma='expy',
    betagammabeta='expy'
)

rate_corr = scaler['c2'][0] / (scaler['c2'][1] / 1000) / np.sum(H)

hist = gvar.gvar(H, np.sqrt(H))

for key in cuts.keys():
    print('_____________{}_____________'.format(key))
    bkg = bkgs[key]
    cut = fit_peak.cut_2d(bins, bins, *cuts[key]) & (H >= 5)
    outputs, inputs, full = fit_peak.fit_peak_2d(bins, bins, hist, cut=cut, bkg=bkg, print_info=1, ax_2d=ax_c2, ax_2d_diff=None if key == 'betagammabeta' else ax_diff, plot_cut=False, corr=key == 'betabeta', full_output=True)
    fit = full['fit']
    norm[key] = outputs['norm'] / (bins[1] - bins[0]) ** 2 * rate_corr
    count[key] = outputs['norm'] / (bins[1] - bins[0]) ** 2
    
    # results table entry
    if bkg == 'expcross':
        exp_ampl_0   = fmt(outputs['exp_ampl'][0]  )
        exp_ampl_1   = fmt(outputs['exp_ampl'][1]  )
        exp_lambda_0 = fmt(outputs['exp_lambda'][0])
        exp_lambda_1 = fmt(outputs['exp_lambda'][1])
    elif bkg == 'expx':
        exp_ampl_0   = fmt(outputs['exp_ampl']  )
        exp_lambda_0 = fmt(outputs['exp_lambda'])
        exp_ampl_1   = nothing
        exp_lambda_1 = nothing
    elif bkg == 'expy':
        exp_ampl_1   = fmt(outputs['exp_ampl']  )
        exp_lambda_1 = fmt(outputs['exp_lambda'])
        exp_ampl_0   = nothing
        exp_lambda_0 = nothing
    else:
        exp_ampl_0   = nothing
        exp_lambda_0 = nothing
        exp_ampl_1   = nothing
        exp_lambda_1 = nothing
    peak_2d_fit_table.append([
        labels[key],
        fmt(outputs['norm']),
        fmt(outputs['mean'][0]),
        fmt(outputs['sigma'][0]),
        exp_ampl_0,
        exp_lambda_0,
        fmt(outputs['mean'][1]),
        fmt(outputs['sigma'][1]),
        exp_ampl_1,
        exp_lambda_1,
        fmt(outputs['corr']) if 'corr' in outputs else nothing,
        '{:.1f}'.format(fit.chi2),
        '{:d}'.format(fit.dof)
    ])

##### fit 1d

peak_1d_fit_table = []

cuts = {
    ('beta', 1): [250, 370],
    ('gamma', 1): [690, 870],
    ('beta', 2): [200, 300],
    ('gamma', 2): [620, 750]
}

labels.update({key: '$\\{}_{}$'.format(*key) for key in cuts})

for key in cuts:
    print('_____________{}{}_____________'.format(key[0], key[1]))
    h = [H1, H2][key[1] - 1]
    hist = gvar.gvar(h, np.sqrt(h))
    cut = fit_peak.cut(bins, cuts[key]) & (h >= 5)
    kw = dict(cut=cut, bkg='exp', print_info=1, ax=[ax_ch1, ax_ch2][key[1] - 1], plot_kw=dict(scaley=False, color='black', label=None), full_output=True)
    if key[0] == 'beta':
        outputs, inputs, full = fit_peak.fit_peak(bins, hist, npeaks=2, **kw)
        this_norm = outputs['peak1_norm'] + outputs['peak2_norm']
    else:
        outputs, inputs, full = fit_peak.fit_peak(bins, hist, npeaks=1, **kw)
        this_norm = outputs['peak1_norm']
    fit = full['fit']
    rate_corr = scaler['ch' + str(key[1])][0] / (scaler['ch' + str(key[1])][1] / 1000) / np.sum(h)
    count[key] = this_norm / (bins[1] - bins[0])
    norm[key] = this_norm / (bins[1] - bins[0]) * rate_corr
    
    # results table entry
    peak_1d_fit_table.append([
        labels[key],
        fmt(this_norm),
        fmt(outputs['peak1_mean']),
        fmt(outputs['peak1_sigma']),
        fmt(outputs['exp_ampl']),
        fmt(outputs['exp_lambda']),
        fmt(outputs['peak2_mean']) if key[0] == 'beta' else nothing,
        fmt(outputs['peak2_sigma']) if key[0] == 'beta' else nothing,
        '{:.1f}'.format(fit.chi2),
        '{:d}'.format(fit.dof)
    ])

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
acc12 = 2 * copy.copy(acc1)

################# fit con 1 != 2

ratios = {
    'p_beta1_tot/p_beta1': 1 / gvar.gvar(0.50, 0.02) * gvar.gvar(1, 0.15), # dal Knoll
    'p_gamma1_tot/p_gamma1': 1 / gvar.gvar(0.22, 0.02) * gvar.gvar(1, 0.15), # dal Knoll
    'p_beta2_tot/p_beta2': 1 / gvar.gvar(0.50, 0.02) * gvar.gvar(1, 0.15), # dal Knoll
    'p_gamma2_tot/p_gamma2': 1 / gvar.gvar(0.22, 0.02) * gvar.gvar(1, 0.15) # dal Knoll
}

ratios['p_beta12_tot/p_beta12'] = ratios['p_beta1_tot/p_beta1'] + ratios['p_beta2_tot/p_beta2'] - 1

Y = {}
Y.update(norm)
Y.update(ratios)
labels.update({key: '${}$'.format(key.replace('beta', '{\\beta').replace('gamma', '{\\gamma').replace('1', '1}').replace('2', '2}').replace('1}2}', '12}').replace('_tot', '\\tot')) for key in ratios})

p0 = dict(
    Rp_beta12=norm['betabeta'],
    Rp_beta1=norm['beta', 1] / 2,
    Rp_beta2=norm['beta', 2] / 2,
    p_gamma1=norm['gammabeta'] / norm['beta', 2],
    p_gamma2=norm['betagamma'] / norm['beta', 1],
    R_tot=norm['gamma', 1] * norm['beta', 2] / norm['gammabeta']
)

p0.update(dict(
    Rp_beta1_tot =  Y['p_beta1_tot/p_beta1'  ] * p0['Rp_beta1' ],
    p_gamma1_tot =  Y['p_gamma1_tot/p_gamma1'] * p0['p_gamma1' ],
    Rp_beta2_tot =  Y['p_beta2_tot/p_beta2'  ] * p0['Rp_beta2' ],
    p_gamma2_tot =  Y['p_gamma2_tot/p_gamma2'] * p0['p_gamma2' ],
    Rp_beta12_tot = Y['p_beta12_tot/p_beta12'] * p0['Rp_beta12']
))

def fcn(p):
    ans = {}

    R_tot = p['R_tot']
    Rp_beta1 = p['Rp_beta1']
    Rp_beta2 = p['Rp_beta2']
    p_gamma1 = p['p_gamma1']
    p_gamma2 = p['p_gamma2']
    Rp_beta12 = p['Rp_beta12']
    Rp_beta1_tot = p['Rp_beta1_tot']
    Rp_beta2_tot = p['Rp_beta2_tot']
    p_gamma1_tot = p['p_gamma1_tot']
    p_gamma2_tot = p['p_gamma2_tot']
    Rp_beta12_tot = p['Rp_beta12_tot']

    ans['p_beta1_tot/p_beta1'] = Rp_beta1_tot / Rp_beta1
    ans['p_gamma1_tot/p_gamma1'] = p_gamma1_tot / p_gamma1
    ans['p_beta2_tot/p_beta2'] = Rp_beta2_tot / Rp_beta2
    ans['p_gamma2_tot/p_gamma2'] = p_gamma2_tot / p_gamma2
    ans['p_beta12_tot/p_beta12'] = Rp_beta12_tot / Rp_beta12

    ans['beta', 1] = 2 * Rp_beta1 * (1 - p_gamma1_tot)
    ans['gamma', 1] = R_tot * p_gamma1 - 2 * Rp_beta1_tot * p_gamma1
    ans['gammabeta'] = 2 * Rp_beta2 * p_gamma1 - Rp_beta12_tot * p_gamma1
    ans['gammabetabeta'] = Rp_beta12 * p_gamma1

    ans['beta', 2] = 2 * Rp_beta2 * (1 - p_gamma2_tot)
    ans['gamma', 2] = R_tot * p_gamma2 - 2 * Rp_beta2_tot * p_gamma2
    ans['betagamma'] = 2 * Rp_beta1 * p_gamma2 - Rp_beta12_tot * p_gamma2
    ans['betagammabeta'] = Rp_beta12 * p_gamma2

    ans['betabeta'] = Rp_beta12 * (1 - p_gamma1_tot - p_gamma2_tot)

    return ans

fit = lsqfit.nonlinear_fit(data=Y, fcn=fcn, p0=gvar.mean(p0), debug=True)
print(fit.format(maxline=True))

R = acc12 / (acc1 * acc2) * fit.p['Rp_beta1'] * fit.p['Rp_beta2'] / fit.p['Rp_beta12']

################# fit con 1 = 2

# ratios = {
#     # 'p_beta_tot/p_beta': cross.total(0.511) / cross.photoel(0.511) * gvar.gvar(1, 0.2), # calcolare meglio
#     # 'p_gamma_tot/p_gamma': cross.total(1.275) / cross.photoel(1.275) * gvar.gvar(1, 0.2), # calcolare meglio
#     'p_beta_tot/p_beta': 1 / gvar.gvar(0.50, 0.02) * gvar.gvar(1, 0.1), # dal Knoll
#     'p_gamma_tot/p_gamma': 1 / gvar.gvar(0.22, 0.02) * gvar.gvar(1, 0.1) # dal Knoll
# }
#
# ratios['p_beta12_tot/p_beta12'] = 2 * ratios['p_beta_tot/p_beta'] - 1
#
# Y = {}
# Y.update(norm)
# Y.update(ratios)
#
# p0 = dict(
#     Rp_beta12=norm['betabeta'],
#     Rp_beta=norm['beta', 1] / 2,
#     p_gamma=norm['gammabeta'] / norm['beta', 2],
#     R_tot=norm['gamma', 1] * norm['beta', 2] / norm['gammabeta']
# )
#
# p0.update(dict(
#     Rp_beta_tot =  ratios['p_beta_tot/p_beta'  ] * p0['Rp_beta' ],
#     p_gamma_tot =  ratios['p_gamma_tot/p_gamma'] * p0['p_gamma' ],
#     Rp_beta12_tot = ratios['p_beta12_tot/p_beta12'] * p0['Rp_beta12']
# ))
#
# def fcn(p):
#     ans = {}
#
#     R_tot = p['R_tot']
#     Rp_beta = p['Rp_beta']
#     p_gamma = p['p_gamma']
#     Rp_beta12 = p['Rp_beta12']
#     Rp_beta_tot = p['Rp_beta_tot']
#     p_gamma_tot = p['p_gamma_tot']
#     Rp_beta12_tot = p['Rp_beta12_tot']
#
#     ans['p_beta_tot/p_beta'] = Rp_beta_tot / Rp_beta
#     ans['p_gamma_tot/p_gamma'] = p_gamma_tot / p_gamma
#     ans['p_beta12_tot/p_beta12'] = Rp_beta12_tot / Rp_beta12
#
#     ans['beta', 1] = 2 * Rp_beta * (1 - p_gamma_tot)
#     ans['gamma', 1] = R_tot * p_gamma - 2 * Rp_beta_tot * p_gamma
#     ans['gammabeta'] = 2 * Rp_beta * p_gamma - Rp_beta12_tot * p_gamma
#     ans['gammabetabeta'] = Rp_beta12 * p_gamma
#
#     ans['beta', 2] = 2 * Rp_beta * (1 - p_gamma_tot)
#     ans['gamma', 2] = R_tot * p_gamma - 2 * Rp_beta_tot * p_gamma
#     ans['betagamma'] = 2 * Rp_beta * p_gamma - Rp_beta12_tot * p_gamma
#     ans['betagammabeta'] = Rp_beta12 * p_gamma
#
#     ans['betabeta'] = Rp_beta12 * (1 - 2 * p_gamma_tot)
#
#     return ans
#
# fit = lsqfit.nonlinear_fit(data=Y, fcn=fcn, p0=gvar.mean(p0), debug=True)
# print(fit.format(maxline=True))
#
# R = acc12 / (acc1 * acc2) * fit.p['Rp_beta'] ** 2 / fit.p['Rp_beta12']

#################

# print results

rat = R / fit.p['R_tot']

input_var = {}
input_var.update(data=list(norm.values()), ratios=list(ratios.values()), accs=[acc1, acc2, acc12])
output_var = dict(R=R, Rtot=fit.p['R_tot'], R_Rtot=rat)

print(gvar.tabulate(output_var))
print(gvar.fmt_errorbudget(output_var, input_var, percent=False))

# format results for report

print('______________peak 2d fit table_______________')
print(lab.TextMatrix(peak_2d_fit_table).transpose().latex())

print('______________peak 1d fit table_______________')
print(lab.TextMatrix(peak_1d_fit_table).transpose().latex())

print('_______________Y fit table________________')
rate_fit_table = []
Y_fitted = fcn(fit.p)
new_labels = {}
for key in labels:
    if key in ratios:
        new_labels[key] = labels[key]
    else:
        new_labels[key] = '$R_{%s}$' % (labels[key].split('$')[1],)
for key in Y:
    rate_fit_table.append([
        new_labels[key],
        fmt(Y[key]),
        fmt(Y_fitted[key])
    ])
print(lab.TextMatrix(rate_fit_table).latex(newline=' & \n'))

print('______________parameters fit table________________')
par_fit_table = []
par_labels = {key: '${}$'.format(key.replace('Rp', '\\R p').replace('R_tot', '\\Rtot').replace('beta', '{\\beta').replace('gamma', '{\\gamma').replace('1', '1}').replace('2', '2}').replace('1}2}', '12}').replace('_tot', '\\tot')) for key in fit.p}
for key in fit.p:
    par_fit_table.append([
        par_labels[key],
        fmt(fit.p[key])
    ])
print(lab.TextMatrix(par_fit_table).latex())

fig.show()
fig_diff.show()
