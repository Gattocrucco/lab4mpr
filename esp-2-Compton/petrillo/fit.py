import numpy as np
import mc9
import empirical
import matplotlib.pyplot as plt
import histo
import lab
import sympy as sp

theta_0s =   [15                           , 0                              , 61.75                              , 45]
files =      ['../dati/histo-27feb-e15.dat', '../dati/histo-23feb-notte.dat', '../dati/histo-22feb-stralunga.dat', '../dati/histo-20feb-notte.dat']
calib_date = ['27feb'                      , '22feb'                        , '22feb'                            , '22feb']
fitcuts =    [(3000, 8000)                 , (3000, 8000)                   , (1500, 4500)                       , (2000, 6000)]

# def hist_adc(a, weights=None):
# 	return empirical.histogram(a, bins=2**13, range=(0, 2**13), weights=weights)

fig = plt.figure('fit')
fig.clf()

for i in range(len(files)):
	filename = files[i]
	theta_0 = theta_0s[i]
	fitcut = fitcuts[i]
	
	print('FILE {}'.format(filename))
	print('loading data...')
	counts = np.loadtxt(filename, unpack=True)

	print('monte carlo...')
	pa, sa, wpa, wsa = mc9.mc_cached(1.33, theta_0=theta_0, N=1000000, seed=0, date=calib_date[i])
	pb, sb, wpb, wsb = mc9.mc_cached(1.17, theta_0=theta_0, N=1000000, seed=1, date=calib_date[i])
	wsa /= 7
	wsb /= 7

	print('empirical...')
	empa = empirical.EmpiricalSecondary(sa, wsa, symb=True)
	empb = empirical.EmpiricalSecondary(sb, wsb, symb=True)

	def fit_fun_a(e, N1, mu1, sigma1, Ns1, scale1, N2, mu2, sigma2, Ns2, scale2, ampl, tau):
	    gaus1 = N1 / (sp.sqrt(2 * np.pi) * sigma1) * sp.exp(-1/2 * (e - mu1)**2 / sigma1**2)
	    sh1 = Ns1 * empa(e, scale1)
	    return gaus1 + sh1
    
	def fit_fun_b(e, N1, mu1, sigma1, Ns1, scale1, N2, mu2, sigma2, Ns2, scale2, ampl, tau):
	    gaus2 = N2 / (sp.sqrt(2 * np.pi) * sigma2) * sp.exp(-1/2 * (e - mu2)**2 / sigma2**2)
	    sh2 = Ns2 * empb(e, scale2)
	    return gaus2 + sh2

	def fit_fun_c(e, N1, mu1, sigma1, Ns1, scale1, N2, mu2, sigma2, Ns2, scale2, ampl, tau):
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
	total = np.sum(counts) * rebin
	mc_total = np.sum(wpa) + np.sum(wsa) + np.sum(wpb) + np.sum(wsb)
	ratio = total / mc_total
	p0 = [np.sum(wpa) * ratio, np.mean(pa), np.std(pa), np.sum(wsa) * ratio, 1, np.sum(wpb) * ratio, np.mean(pb), np.std(pb), np.sum(wsb) * ratio, 1, np.max(counts[100:len(counts) // 2]), 1500]
	bounds = [
		 [0, -np.inf, 0, 0, 0, 0, -np.inf, 0, 0, 0, 0, 0],
		 [np.inf] * len(p0)
	]

	# for i in range(len(p0)):
	#     p0[i] = np.random.uniform(0.9 * p0[i], 1.1 * p0[i])

	model = lab.CurveModel(fit_fun, symb=True, npar=len(p0))
	try:
		out = lab.fit_curve(model, fit_x, fit_y, dy=fit_dy, p0=p0, print_info=1, bounds=bounds)
		par = out.par
	except RuntimeError:
		par = p0
		print('fit failed!')

	modela = lab.CurveModel(fit_fun_a, symb=True)
	modelb = lab.CurveModel(fit_fun_b, symb=True)
	modelc = lab.CurveModel(fit_fun_c, symb=True)

	ax = fig.add_subplot(len(files) // 2, 2, i + 1)
	histo.bar_line(edges, histo.partial_sum(counts, rebin), ax=ax, label=filename.split('/')[-1])
	ax.plot(fit_x,  model.f()(fit_x, *p0), '-k')
	ax.plot(fit_x,  model.f()(fit_x, *par), '-r')
	ax.plot(fit_x, modela.f()(fit_x, *par), '--r', linewidth=0.5)
	ax.plot(fit_x, modelb.f()(fit_x, *par), '--r', linewidth=0.5)
	ax.plot(fit_x, modelc.f()(fit_x, *par), '--r', linewidth=0.5)
	ax.grid()
	ax.legend(fontsize='small', loc=2)

fig.show()
