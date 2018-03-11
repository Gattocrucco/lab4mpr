import numpy as np
import mc9
import empirical
import matplotlib.pyplot as plt
import histo
import lab
import sympy as sp
from uncertainties import unumpy as unp
import uncertainties as un
import collections
import pickle

theta_0s   = [15                         , 15                         , 15                         , 15                         , 15                         , 15                         , 7                     , 7                     , 7                     , 61.75                              , 45                             ]
files      = ['../dati/log-27feb-e15.npy', '../dati/log-27feb-e15.npy', '../dati/log-27feb-e15.npy', '../dati/log-27feb-e15.npy', '../dati/log-27feb-e15.npy', '../dati/log-27feb-e15.npy', '../dati/log-neve.npy', '../dati/log-neve.npy', '../dati/log-neve.npy', '../dati/histo-22feb-stralunga.dat', '../dati/histo-20feb-notte.dat']
logcut     = [(0, 1/2)                   , (0, 1/4)                   , (1/4, 1/2)                 , (1/2, 1)                   , (1/2, 3/4)                 , (3/4, 1)                   , (0, 1/5)              , (0, 1/10)             , (1/10, 1/5)           , None                               , None                           ]
calib_date = ['26feb'                    , '26feb'                    , '26feb'                    , '27feb'                    , '27feb'                    , '27feb'                    , '27feb'               , '27feb'               , '27feb'               , '22feb'                            , '20feb'                        ]
fitcuts    = [(3000, 7200)               , (3000, 7200)               , (3000, 7200)               , (3000, 7200)               , (3000, 7200)               , (3000, 7200)               , (3000, 7400)          , (3000, 7400)          , (3000, 7400)          , (1500, 3700)                       , (2000, 5100)                   ]
Ls         = [40                         , 40                         , 40                         , 40                         , 40                         , 40                         , 40                    , 40                    , 40                    , 71.5 + 62.8 - 16                   , 40                             ]
fixnorm    = [False                      , False                      , False                      , False                      , False                      , False                      , False                 , False                 , False                 , True                               , True                           ]
labels     = ['15°, prima metà'          , '15°, prima metà'          , '15°, prima metà'          , '15°, seconda metà'        , '15°, seconda metà'        , '15°, seconda metà'        , '7°, primo quinto'    , '7°, primo quinto'    , '7°, primo quinto'    , '61.75°'                           , '45°'                          ]
bkg_sim    = [False                      , False                      , False                      , False                      , False                      , False                      , False                 , False                 , False                 , True                               , False                          ]
doplot     = [True                       , False                      , False                      , True                       , False                      , False                      , True                  , False                 , False                 , True                               , True                           ]

# def hist_adc(a, weights=None):
# 	return empirical.histogram(a, bins=2**13, range=(0, 2**13), weights=weights)

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

fig = plt.figure('fit')
fig.clf()
fig.set_tight_layout(True)

centers_133 = []
centers_117 = []
centers_133_sim = []
centers_117_sim = []

for i in range(len(files)):
    filename = files[i]
    theta_0 = theta_0s[i]
    fitcut = fitcuts[i]
    
    print('FILE {}'.format(filename))
    
    print('loading data...')
    if filename.endswith('.dat'):
        counts = np.loadtxt(filename, unpack=True)
    elif filename.endswith('.npy'):
        samples = np.load(filename)
        if not (logcut is None):
            cutl = int(np.floor(logcut[i][0] * len(samples)))
            cutr = int(np.floor(logcut[i][1] * len(samples)))
            samples = samples[cutl:cutr]
        counts = np.bincount(samples, minlength=2**13)

    print('monte carlo...')
    pa, sa, wpa, wsa = mc9.mc_cached(1.33, theta_0=theta_0, N=1000000, seed=0, nai_distance=Ls[i], date=calib_date[i])
    pb, sb, wpb, wsb = mc9.mc_cached(1.17, theta_0=theta_0, N=1000000, seed=1, nai_distance=Ls[i], date=calib_date[i])
    wsa /= 7
    wsb /= 7

    print('empirical...')
    empa = empirical.EmpiricalSecondary(sa, wsa, symb=True)
    empb = empirical.EmpiricalSecondary(sb, wsb, symb=True)

    def fit_fun_a(e, N1, mu1, sigma1, Ns1, scale1, f2, mu1_mu2, sigma2, Ns2, scale2, ampl, tau):
        gaus1 = N1 / (sp.sqrt(2 * np.pi) * sigma1) * sp.exp(-1/2 * (e - mu1)**2 / sigma1**2)
        sh1 = Ns1 * empa(e, scale1)
        return gaus1 + sh1
    
    def fit_fun_b(e, N1, mu1, sigma1, Ns1, scale1, f2, mu1_mu2, sigma2, Ns2, scale2, ampl, tau):
        gaus2 = N1 * f2 / (sp.sqrt(2 * np.pi) * sigma2) * sp.exp(-1/2 * (e - mu1 + mu1_mu2)**2 / sigma2**2)
        sh2 = Ns2 * empb(e, scale2)
        return gaus2 + sh2

    def fit_fun_c(e, N1, mu1, sigma1, Ns1, scale1, f2, mu1_mu2, sigma2, Ns2, scale2, ampl, tau):
        return ampl * sp.exp(-e / tau)

    def fit_fun(e, *p):
        return fit_fun_a(e, *p) + fit_fun_b(e, *p) + fit_fun_c(e, *p)

    print('fit...')
    # prepare data for fit
    rebin = 8
    edges = np.arange(2**13 + 1)[::rebin]
    cut = (edges[:-1] >= fitcut[0]) & (edges[:-1] <= fitcut[1])
    fit_x = edges[:-1][cut] + rebin / 2
    fit_y = histo.partial_sum(counts, rebin)[cut]
    fit_dy = np.where(fit_y > 0, np.sqrt(fit_y), 1)

    # estimate initial parameters
    total = np.sum(counts[100:8000]) * rebin
    mc_total = np.sum(wpa) + np.sum(wsa) + np.sum(wpb) + np.sum(wsb)
    ratio = total / mc_total
    p0 = [np.sum(wpa) * ratio, np.mean(pa), np.std(pa), np.sum(wsa) * ratio, 1, np.sum(wpb) * ratio, np.mean(pb), np.std(pb), np.sum(wsb) * ratio, 1, np.max(counts[100:len(counts) // 2]), 1500]
    p0[5] /= p0[0]
    p0[6] = p0[1] - p0[6]
    pfix = np.zeros(len(p0), dtype=bool)
    if fixnorm[i]:
        pfix[5] = True
    bounds = [
         [0, -np.inf, 0, 0, 0, 0, 0, 0, 0, 0, 0, 100],
         [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf]
    ]

    # for i in range(len(p0)):
    #     p0[i] = np.random.uniform(0.9 * p0[i], 1.1 * p0[i])

    model = lab.CurveModel(fit_fun, symb=True, npar=len(p0))
    if not bkg_sim[i]:
        try:
            out = lab.fit_curve(model, fit_x, fit_y, dy=fit_dy, p0=p0, pfix=pfix, print_info=3, method='linodr', bounds=bounds)
            par = out.par
        except:
            par = p0
            print('fit failed!')
    
    print('simplified background fit...')
    p0_sim = np.copy(p0)
    pfix = [3, 8]
    p0_sim[pfix] = 0
    if fixnorm[i]:
        pfix += [5]
    pfix += [4, 9]
    # try:
    out_sim = lab.fit_curve(model, fit_x, fit_y, dy=fit_dy, p0=p0_sim, pfix=pfix, print_info=3, method='linodr', bounds=bounds)
    par_sim = out_sim.par
    # except:
    #     par_sim = p0_sim
    #     print('fit failed!')
    
    print('plot...')
    if doplot[i]:
        lplot = np.sum(doplot)
        iplot = np.cumsum(doplot)[i] - 1
        
        modela = lab.CurveModel(fit_fun_a, symb=True)
        modelb = lab.CurveModel(fit_fun_b, symb=True)
        modelc = lab.CurveModel(fit_fun_c, symb=True)
    
        ncols = 2
        nrows = (lplot + 1) // ncols + (1 if (lplot + 1) % ncols else 0)
        ax = fig.add_subplot(nrows, ncols, iplot + 1)
        rebin_counts = histo.partial_sum(counts, rebin)
        color3 = [0.8] * 3
        histo.bar_line(edges, rebin_counts, ax=ax, label=labels[i], color=color3)
        # ax.plot(fit_x,  model.f()(fit_x, *p0), '-k')
        # ax.plot(fit_x, modela.f()(fit_x, *p0), '--k', linewidth=0.5)
        # ax.plot(fit_x, modelb.f()(fit_x, *p0), '--k', linewidth=0.5)
        # ax.plot(fit_x, modelc.f()(fit_x, *p0), '--k', linewidth=0.5)
        color = [0.0] * 3
        if not bkg_sim[i]:
            ax.plot(fit_x,  model.f()(fit_x, *par), '-', color=color)
            ax.plot(fit_x, modela.f()(fit_x, *par), '--', color=color, linewidth=0.5)
            ax.plot(fit_x, modelb.f()(fit_x, *par), '--', color=color, linewidth=0.5)
            ax.plot(fit_x, modelc.f()(fit_x, *par), '--', color=color, linewidth=0.5)
        color2 = [0.4] * 3
        ax.plot(fit_x,  model.f()(fit_x, *par_sim), '-', color=color2)
        if bkg_sim[i]:
            ax.plot(fit_x, modela.f()(fit_x, *par_sim), '-.', linewidth=0.5, color=color2)
            ax.plot(fit_x, modelb.f()(fit_x, *par_sim), '-.', linewidth=0.5, color=color2)
            ax.plot(fit_x, modelc.f()(fit_x, *par_sim), '-.', linewidth=0.5, color=color2)
        ax.grid(linestyle=':')
        ax.legend(fontsize='small', loc=2)
        ymax = np.max(rebin_counts[15:1000])
        ax.set_ylim((-ymax * 0.05, ymax * 1.05))
        if iplot % ncols == 0:
            ax.set_ylabel('conteggi')
        if lplot - iplot <= 2:
            ax.set_xlabel('canale ADC')
    
        if iplot == lplot - 1:
            ax = fig.add_subplot(nrows, ncols, nrows * ncols)
            dummy = ([0], [0])
            ax.plot(*dummy, '-', color=color3, label='dati')
            ax.plot(*dummy, '-', color=color, label='fit')
            ax.plot(*dummy, '--', linewidth=0.5, color=color, label='  addendi')
            ax.plot(*dummy, '-', color=color2, label='fit fondo semplificato')
            if any(bkg_sim):
                ax.plot(*dummy, '-.', color=color2, label='  addendi', linewidth=0.5)
            ax.set_xlim((10,20))
            ax.set_ylim((10,20))
            ax.axis('off')
            ax.legend(loc='center')

    idx = np.array([1,6])
    if not bkg_sim[i]:
        c133, c133_c117 = un.correlated_values(out.par[idx], out.cov[np.ix_(idx,idx)], tags=['fit'] * 2)
        c117 = c133 - c133_c117
    else:
        c133, c117 = np.nan, np.nan
    centers_133.append(c133)
    centers_117.append(c117)
    c133_sim, c133_c117_sim = un.correlated_values(out_sim.par[idx], out_sim.cov[np.ix_(idx,idx)], tags=['fit'] * 2)
    c117_sim = c133_sim - c133_c117_sim
    centers_133_sim.append(c133_sim)
    centers_117_sim.append(c117_sim)

print('saving to fit-result.pickle...')
with open('fit-result.pickle', 'wb') as dump_file:
    pickle.dump([centers_133, centers_117, centers_133_sim, centers_117_sim, theta_0s, calib_date, fixnorm], dump_file)

fig.show()
