import numpy as np
import mc9
import empirical
import matplotlib.pyplot as plt
import histo
import lab
import sympy as sp
from uncertainties import unumpy as unp
import uncertainties as un
import calibration
import collections
import bias

theta_0s =   [15                         , 15                         , 7                     , 61.75                              , 45]
files =      ['../dati/log-27feb-e15.npy', '../dati/log-27feb-e15.npy', '../dati/log-neve.npy', '../dati/histo-22feb-stralunga.dat', '../dati/histo-20feb-notte.dat']
logcut     = [(0, 1/2)                   , (1/2, 1)                   , (0, 1/5)              , None                               , None                           ]
calib_date = ['26feb'                    , '27feb'                    , '27feb'               , '22feb'                            , '20feb']
fitcuts =    [(3000, 8000)               , (3000, 8000)               , (3000, 8000)          , (1500, 4500)                       , (2000, 6000)]
Ls         = [40                         , 40                         , 40                    , 71.5 + 62.8 - 16                   , 40]
fixnorm    = [False                      , False                      , False                 , True                               , True]

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

	def fit_fun_a(e, N1, mu1, sigma1, Ns1, scale1, f2, mu2, sigma2, Ns2, scale2, ampl, tau):
	    gaus1 = N1 / (sp.sqrt(2 * np.pi) * sigma1) * sp.exp(-1/2 * (e - mu1)**2 / sigma1**2)
	    sh1 = Ns1 * empa(e, scale1)
	    return gaus1 + sh1
    
	def fit_fun_b(e, N1, mu1, sigma1, Ns1, scale1, f2, mu2, sigma2, Ns2, scale2, ampl, tau):
	    gaus2 = N1 * f2 / (sp.sqrt(2 * np.pi) * sigma2) * sp.exp(-1/2 * (e - mu2)**2 / sigma2**2)
	    sh2 = Ns2 * empb(e, scale2)
	    return gaus2 + sh2

	def fit_fun_c(e, N1, mu1, sigma1, Ns1, scale1, f2, mu2, sigma2, Ns2, scale2, ampl, tau):
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
	pfix = np.zeros(len(p0), dtype=bool)
	if fixnorm[i]:
		pfix[5] = True
	bounds = [
		 [0, -np.inf, 0, 0, 0, 0, -np.inf, 0, 0, 0, 0, 100],
		 [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf]
	]

	# for i in range(len(p0)):
	#     p0[i] = np.random.uniform(0.9 * p0[i], 1.1 * p0[i])

	model = lab.CurveModel(fit_fun, symb=True, npar=len(p0))
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
	# 	par_sim = p0_sim
	# 	print('fit failed!')

	print('plot...')
	modela = lab.CurveModel(fit_fun_a, symb=True)
	modelb = lab.CurveModel(fit_fun_b, symb=True)
	modelc = lab.CurveModel(fit_fun_c, symb=True)

	ax = fig.add_subplot(len(files) // 2 + (1 if len(files) % 2 else 0), 2, i + 1)
	rebin_counts = histo.partial_sum(counts, rebin)
	histo.bar_line(edges, rebin_counts, ax=ax, label=filename.split('/')[-1])
	ax.plot(fit_x,  model.f()(fit_x, *p0), '-k')
	ax.plot(fit_x, modela.f()(fit_x, *p0), '--k', linewidth=0.5)
	ax.plot(fit_x, modelb.f()(fit_x, *p0), '--k', linewidth=0.5)
	ax.plot(fit_x, modelc.f()(fit_x, *p0), '--k', linewidth=0.5)
	ax.plot(fit_x,  model.f()(fit_x, *par), '-r')
	ax.plot(fit_x, modela.f()(fit_x, *par), '--r', linewidth=0.5)
	ax.plot(fit_x, modelb.f()(fit_x, *par), '--r', linewidth=0.5)
	ax.plot(fit_x, modelc.f()(fit_x, *par), '--r', linewidth=0.5)
	ax.plot(fit_x,  model.f()(fit_x, *par_sim), '-g')
	ax.plot(fit_x, modela.f()(fit_x, *par_sim), '--g', linewidth=0.5)
	ax.plot(fit_x, modelb.f()(fit_x, *par_sim), '--g', linewidth=0.5)
	ax.plot(fit_x, modelc.f()(fit_x, *par_sim), '--g', linewidth=0.5)
	ax.grid()
	ax.legend(fontsize='small', loc=2)
	ymax = np.max(rebin_counts[15:1000])
	ax.set_ylim((-ymax * 0.05, ymax * 1.05))
	
	idx = np.array([1,6])
	c133, c117 = un.correlated_values(out.par[idx], out.cov[np.ix_(idx,idx)], tags=['fit'] * 2)
	centers_133.append(c133)
	centers_117.append(c117)
	c133_sim, c117_sim = un.correlated_values(out_sim.par[idx], out_sim.cov[np.ix_(idx,idx)], tags=['fit'] * 2)
	centers_133_sim.append(c133_sim)
	centers_117_sim.append(c117_sim)
	
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

fig2 = plt.figure('fit_result')
fig2.clf()
ax = fig2.add_subplot(111)

ax.errorbar(np.arange(len(theta_0s)) - 0.05, unp.nominal_values(m_133), yerr=unp.std_devs(m_133), fmt='.', label='1.33')
ax.errorbar(np.arange(len(theta_0s)) + 0.05, unp.nominal_values(m_117), yerr=unp.std_devs(m_117), fmt='.', label='1.17')
ax.errorbar(np.arange(len(theta_0s)) - 0.1, unp.nominal_values(m_133_sim), yerr=unp.std_devs(m_133_sim), fmt='.', label='1.33 simp.')
ax.errorbar(np.arange(len(theta_0s)) + 0.1, unp.nominal_values(m_117_sim), yerr=unp.std_devs(m_117_sim), fmt='.', label='1.17 simp.')
ax.set_xticks(np.arange(len(theta_0s)))
ax.set_xticklabels(theta_0s)
ax.legend(loc=1)

fig.show()
fig2.show()
