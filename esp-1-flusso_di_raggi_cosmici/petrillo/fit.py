import numpy as np
import montecarlo as mc
import uncertainties as un
from uncertainties import unumpy as unp
from uncertainties import umath
import copy
from collections import OrderedDict
from scipy import optimize, linalg, interpolate
import time
from matplotlib import pyplot as plt
from matplotlib import gridspec, patches
import sys
import lab

####### things #######

def errorsummary(x):
    comps = x.error_components()
    
    tags = set(map(lambda v: v.tag, comps.keys()))
    var = dict(zip(tags, [0] * len(tags)))
    
    for (v, sd) in comps.items():
        var[v.tag] += sd ** 2
    
    tags = list(tags)
    sds = np.sqrt(np.array([var[tag] for tag in tags]))
    idx = np.argsort(sds)[::-1]
    d = OrderedDict()
    for i in idx:
        d[tags[i]] = sds[i]
    
    return d

def clear_lines(nlines, nrows):
    for i in range(nlines):
        sys.stdout.write('\033[F\r%s\r' % (" " * nrows,))
    sys.stdout.flush()

####### load data #######

def loadtxtlbs(filename, labels, prefit=False):
    data = np.loadtxt(filename, unpack=True)
    datadict = dict()
    if len(data) != len(labels):
        raise RuntimeError('length of data %d != length of labels %d' % (len(data), len(labels)))
    for i in range(len(data)):
        datadict[labels[i]] = data[i]
    p = datadict['prefit']
    c = p == prefit
    for k in datadict.keys():
        if k != 'prefit':
            datadict[k] = datadict[k][c]
    datadict['prefit'] = p[c]
    return datadict

data2lbs = ['sogliaA', 'sogliaB', 'alimA', 'alimB', 'clock', 'A', 'B', 'A&B', 'a&c', 'a&b&c', 'a&b&c&A', 'a&b&c&B', 'PMTA', 'PMTB', 'PMTa', 'PMTb', 'PMTc', 'prefit']
data2albs = ['sogliaA', 'sogliaB', 'alimA', 'alimB', 'clock', 'A', 'B', 'A&B', 'a&b', 'a&b&B', 'a&b&A', 'a&b&A&B', 'PMTA', 'PMTB', 'PMTa', 'PMTb', 'prefit']
data3lbs = ['sogliaA', 'sogliaB', 'sogliaC', 'alimA', 'alimB', 'alimC', 'clock', 'C', 'B', 'A&B&C', 'a&b', 'a&b&C', 'a&b&B', 'a&b&A', 'PMTA', 'PMTB', 'PMTC', 'PMTa', 'PMTb', 'prefit']
dataefflbs = ['sogliaA', 'clock', 'A', 'a&b', 'a&b&A', 'b&c', 'b&c&A', 'b&d', 'b&d&A', 'PMTA', 'PMTa', 'PMTb', 'PMTc', 'PMTd', 'prefit']

data2 = loadtxtlbs('fitdata2.txt', data2lbs)
data2a = loadtxtlbs('fitdata2a.txt', data2albs)
data3 = loadtxtlbs('fitdata3.txt', data3lbs)
dataeff = loadtxtlbs('fitdataeff.txt', dataefflbs)

data2sigA = un.ufloat(38, 2, tag='data2sigA') * 1e-9
data2sigB = un.ufloat(37, 2, tag='data2sigB') * 1e-9

data2asigA = data2sigA
data2asigB = data2sigB

time1e5 = un.ufloat(1e5, 0.5, tag='data3time') * 1e-3
data3rateA = un.ufloat(4243753, np.sqrt(4243753), tag='data3countA') / time1e5 # logbook:14dic
data3ratea = un.ufloat(86130, np.sqrt(86130), tag='data3counta') / time1e5
data3rateb = un.ufloat(31799, np.sqrt(31799), tag='data3countb') / time1e5

data3sigA = data2sigA
data3sigB = data2sigB
data3sigC = un.ufloat(35, 1, tag='data3sigC') * 1e-9
data3siga = un.ufloat(35, 1, tag='data3siga') * 1e-9
data3sigb = un.ufloat(35, 1, tag='data3sigb') * 1e-9

dataeffrate = un.ufloat(500, 30, tag='dataeffrate')
dataeffsigA = data3sigA
dataeffsiga = data3siga
dataeffsigb = data3sigB
dataeffsigc = data3sigC
dataeffsigd = data3sigb

####### prepare monte carlo #######

# draw samples
print('drawing samples for acceptances MC...')
mcobj = mc.MC(*[mc.pmt(i+1) for i in range(6)])
mcobj.random_samples(N=300000)

print('drawing samples for geometry uncertainty MC...')
mcgeom = mc.MC(*[mc.pmt(i+1) for i in range(6)])
mcgeom.random_samples(N=10000)
mcgeom.sample_geometry(3000)

# create list of expressions to compute
mclist = []
for i in range(len(data2['clock'])):
    mclist += [
        {data2['PMTA'][i], data2['PMTB'][i]},
        {data2['PMTa'][i], data2['PMTb'][i], data2['PMTc'][i]},
        {data2['PMTa'][i], data2['PMTb'][i], data2['PMTc'][i], data2['PMTA'][i]},
        {data2['PMTa'][i], data2['PMTb'][i], data2['PMTc'][i], data2['PMTB'][i]}
    ]
for i in range(len(data2a['clock'])):
    mclist += [
        {data2a['PMTA'][i], data2a['PMTB'][i]},
        {data2a['PMTa'][i], data2a['PMTb'][i], data2a['PMTA'][i], data2a['PMTB'][i]},
        {data2a['PMTa'][i], data2a['PMTb'][i], data2a['PMTA'][i]},
        {data2a['PMTa'][i], data2a['PMTb'][i], data2a['PMTB'][i]}
    ]
for i in range(len(data3['clock'])):
    mclist += [
        {data3['PMTA'][i], data3['PMTB'][i], data3['PMTC'][i]},
        {data3['PMTa'][i], data3['PMTb'][i]},
        {data3['PMTa'][i], data3['PMTb'][i], data3['PMTA'][i]},
        {data3['PMTa'][i], data3['PMTb'][i], data3['PMTB'][i]},
        {data3['PMTa'][i], data3['PMTb'][i], data3['PMTC'][i]}
    ]
for i in range(len(dataeff['clock'])):
    mclist += [
        {dataeff['PMTa'][i], dataeff['PMTb'][i]},
        {dataeff['PMTb'][i], dataeff['PMTc'][i]},
        {dataeff['PMTb'][i], dataeff['PMTd'][i]},
        {dataeff['PMTa'][i], dataeff['PMTb'][i], dataeff['PMTA'][i]},
        {dataeff['PMTb'][i], dataeff['PMTc'][i], dataeff['PMTA'][i]},
        {dataeff['PMTb'][i], dataeff['PMTd'][i], dataeff['PMTA'][i]}
    ]

# decide pivots
cexprs = dict() # format of MC.count
exprs = dict() # format used as key
while len(mclist) > 0:
    # count how many times each PMT is used in an expression
    count = [0] * 6
    for i in range(6):
        for s in mclist:
            count[i] += (i + 1) in s
    # pivot is the one used the most
    pivot = np.argmax(count)
    # remove from mclist expressions which contain the pivot,
    # and put them in a per-pivot list
    pivotset = set()
    i = 0
    while i < len(mclist):
        if pivot + 1 in mclist[i]:
            pivotset.add(frozenset(map(int, mclist.pop(i))))
        else:
            i += 1
    # convert expressions to the format used by MC.count
    exprs_list = tuple(map(tuple, pivotset))
    cexprs_list = []
    for i in range(len(exprs_list)):
        cexpr = [...] * 6
        for s in exprs_list[i]:
            cexpr[s - 1] = True
        cexprs_list.append(cexpr)
    cexprs[pivot] = cexprs_list
    exprs[pivot] = list(map(frozenset, exprs_list))

####### fit function #######

def f_fit(distr_par, options={}, lamda=1):
    # compute rays
    distr = lambda x: x ** (1 / (1 + distr_par)) # vedi logbook:fit
    mcobj.ray(distr)

    # compute acceptances
    mcexprs = dict()
    for pivot in exprs.keys():
        # run
        mcobj.run(pivot_scint=pivot)  
        # count
        expr = exprs[pivot]
        counts = mcobj.count(*cexprs[pivot], tags=['mc'] * len(cexprs[pivot]))
        # save
        for i in range(len(expr)):
            mcexprs[expr[i]] = counts[i] / mcobj.number_of_rays * mcobj.pivot_horizontal_area

    # compute geometrical uncertainty
    geometry_factors = options.get('geometry_factors', {})
    if len(geometry_factors) == 0:
        mcgeom.ray(distr)
        # compute acceptances for each geometry sample
        mcsegeom = dict()
        for pivot in exprs.keys():
            mcgeom.run(pivot_scint=pivot, randgeom=True)
            counts = mcgeom.count(*cexprs[pivot])
            expr = exprs[pivot]
            for i in range(len(expr)):
                mcsegeom[expr[i]] = counts[i] / mcgeom.number_of_rays * mcgeom.pivot_horizontal_area
        # compute covariance matrix
        mcsegeom_keys = list(mcsegeom.keys())
        mcsegeom_array = np.array([mcsegeom[key] for key in mcsegeom_keys])
        mcsegeom_cov = np.cov(mcsegeom_array)
        assert(len(mcsegeom_cov) == len(mcsegeom_keys))
        # make unitary multiplicative factors containing the geometrical uncertainty
        vals = np.array([mcexprs[key] for key in mcsegeom_keys])
        nom_values = unp.nominal_values(vals)
        geom_factors = np.array(un.correlated_values(nom_values / nom_values, mcsegeom_cov / np.outer(nom_values, nom_values), tags=['geom'] * len(nom_values)))
        for i in range(len(mcsegeom_keys)):
            geometry_factors[mcsegeom_keys[i]] = geom_factors[i]
        options['geometry_factors'] = geometry_factors
    for key in geometry_factors.keys():
        mcexprs[key] *= geometry_factors[key]

    # process data2
    fluxes2 = []
    for i in range(len(data2['clock'])):
        # # assert(data2['prefit'][i])
        time = un.ufloat(data2['clock'][i], 0.5, tag="data2_%dtime" % i) * 1e-3
    
        count = lambda label: un.ufloat(data2[label][i], np.sqrt(data2[label][i]), tag="data2_%dcount%s" % (i, label))
        countA = count('A')
        countB = count('B')
        countAB = count('A&B')
    
        rateA = countA / time
        rateB = countB / time
        rateAB = countAB / time
    
        countabc = count('a&b&c')
        countabcA = count('a&b&c&A')
        countabcB = count('a&b&c&B')
    
        mcAB   = mcexprs[frozenset({data2['PMTA'][i], data2['PMTB'][i]}                                    )]
        mcabc  = mcexprs[frozenset({data2['PMTa'][i], data2['PMTb'][i], data2['PMTc'][i]}                  )]
        mcabcA = mcexprs[frozenset({data2['PMTa'][i], data2['PMTb'][i], data2['PMTc'][i], data2['PMTA'][i]})]
        mcabcB = mcexprs[frozenset({data2['PMTa'][i], data2['PMTb'][i], data2['PMTc'][i], data2['PMTB'][i]})]
    
        noiseAB = lamda * rateA * rateB * (data2sigA + data2sigB)
    
        eff = lambda c3, c2, tag: un.ufloat(c3.n / c2.n, np.sqrt(c3.n/c2.n * (1 - c3.n/c2.n) * 1/c2.n * (1 + 1/c2.n)), tag="data2_%deff%s" % (i, tag))
        effA = eff(countabcA, countabc, 'A') * mcabc / mcabcA
        effB = eff(countabcB, countabc, 'B') * mcabc / mcabcB
    
        flux = (rateAB - noiseAB) / (mcAB * effA * effB)
        fluxes2.append(flux)

    # process data2a
    fluxes2a = []
    for i in range(len(data2a['clock'])):
        # assert(data2a['prefit'][i])
        time = un.ufloat(data2a['clock'][i], 0.5, tag="data2a_%dtime" % i) * 1e-3
    
        count = lambda label: un.ufloat(data2a[label][i], np.sqrt(data2a[label][i]), tag="data2a_%dcount%s" % (i, label))
        countA = count('A')
        countB = count('B')
        countAB = count('A&B')
    
        rateA = countA / time
        rateB = countB / time
        rateAB = countAB / time
    
        countabA = count('a&b&A')
        countabB = count('a&b&B')
        countabAB = count('a&b&A&B')
    
        mcabAB = mcexprs[frozenset({data2a['PMTa'][i], data2a['PMTb'][i], data2a['PMTA'][i], data2a['PMTB'][i]})]
        mcabA = mcexprs[frozenset({data2a['PMTa'][i], data2a['PMTb'][i], data2a['PMTA'][i]})]
        mcabB = mcexprs[frozenset({data2a['PMTa'][i], data2a['PMTb'][i], data2a['PMTB'][i]})]
        mcAB = mcexprs[frozenset({data2a['PMTA'][i], data2a['PMTB'][i]})]
    
        eff = lambda c3, c2, tag: un.ufloat(c3.n / c2.n, np.sqrt(c3.n/c2.n * (1 - c3.n/c2.n) * 1/c2.n * (1 + 1/c2.n)), tag="data2a_%deff%s" % (i, tag))
        # problema: qui le efficienze sono correlate
        effA = eff(countabAB, countabB, 'A') * mcabB / mcabAB
        effB = eff(countabAB, countabA, 'B') * mcabA / mcabAB
    
        noiseAB = lamda * rateA * rateB * (data2asigA + data2asigB)
    
        flux = (rateAB - noiseAB) / (mcAB * effA * effB)
        fluxes2a.append(flux)

    # process data3
    fluxes3 = []
    for i in range(len(data3['clock'])):
        # assert(data3['prefit'][i])
        time = un.ufloat(data3['clock'][i], 0.5, tag="data3_%dtime" % i) * 1e-3
    
        count = lambda label: un.ufloat(data3[label][i], np.sqrt(data3[label][i]), tag="data3_%dcount%s" % (i, label))
        countABC = count('A&B&C')
        countab = count('a&b')
        countabA = count('a&b&A')
        countabB = count('a&b&B')
        countabC = count('a&b&C')
    
        rateABC = countABC / time
        rateB = count('B') / time
        rateC = count('C') / time
        
        noiseab = lamda * data3ratea * data3rateb * (data3siga + data3sigb)
        noiseabA = lamda * data3rateA * noiseab * (data3siga / 2 + data3sigA)
        noiseabB = lamda * rateB      * noiseab * (data3siga / 2 + data3sigB)
        noiseabC = lamda * rateC      * noiseab * (data3siga / 2 + data3sigC)
        noiseABC = lamda**2 * data3rateA * rateB * rateC * (data3sigA + data3sigB) * (data3sigA / 2 + data3sigC)
    
        mcABC = mcexprs[frozenset({data3['PMTA'][i], data3['PMTB'][i], data3['PMTC'][i]})]
        mcab  = mcexprs[frozenset({data3['PMTa'][i], data3['PMTb'][i]}                  )]
        mcabA = mcexprs[frozenset({data3['PMTa'][i], data3['PMTb'][i], data3['PMTA'][i]})]
        mcabB = mcexprs[frozenset({data3['PMTa'][i], data3['PMTb'][i], data3['PMTB'][i]})]
        mcabC = mcexprs[frozenset({data3['PMTa'][i], data3['PMTb'][i], data3['PMTC'][i]})]
    
        def eff(c3, c2, tag):
            e = c3 / c2
            return un.ufloat(e.n, np.sqrt(e.n * (1 - e.n) * 1/c2.n * (1 + 1/c2.n)), tag="data3_%deff%s" % (i, tag)) * e / e.n
        effA = eff(countabA.n - noiseabA * time, countab.n - noiseab * time, 'A') * mcab / mcabA
        effB = eff(countabB.n - noiseabB * time, countab.n - noiseab * time, 'B') * mcab / mcabB
        effC = eff(countabC.n - noiseabC * time, countab.n - noiseab * time, 'C') * mcab / mcabC
    
        flux = (rateABC - noiseABC) / (mcABC * effA * effB * effC)
        fluxes3.append(flux)

    # process dataeff
    efficiencies = []
    for i in range(len(dataeff['clock'])):
        # assert(dataeff['prefit'][i])
        time = un.ufloat(dataeff['clock'][i], 0.5, tag="dataeff_%dtime" % i) * 1e-3
    
        count = lambda label: dataeff[label][i]
        ucount = lambda label: un.ufloat(dataeff[label][i], np.sqrt(dataeff[label][i]), tag="dataeff_%dcount%s" % (i, label))
        countab = count('a&b')
        countabA = count('a&b&A')
        countbc = count('b&c')
        countbcA = count('b&c&A')
        countbd = count('b&d')
        countbdA = count('b&d&A')
    
        ratea = copy.copy(dataeffrate)
        rateb = copy.copy(dataeffrate)
        ratec = copy.copy(dataeffrate)
        rated = copy.copy(dataeffrate)
        rateA = ucount('A') / time
    
        noiseab = lamda * ratea * rateb * (dataeffsiga + dataeffsigb)
        noisebc = lamda * rateb * ratec * (dataeffsigb + dataeffsigc)
        noisebd = lamda * rateb * rated * (dataeffsigb + dataeffsigd)
        noiseabA = lamda * noiseab * rateA * (dataeffsiga / 2 + dataeffsigA)
        noisebcA = lamda * noisebc * rateA * (dataeffsigb / 2 + dataeffsigA)
        noisebdA = lamda * noisebd * rateA * (dataeffsigb / 2 + dataeffsigA)
    
        mcab = mcexprs[frozenset({dataeff['PMTa'][i], dataeff['PMTb'][i]})]
        mcbc = mcexprs[frozenset({dataeff['PMTb'][i], dataeff['PMTc'][i]})]
        mcbd = mcexprs[frozenset({dataeff['PMTb'][i], dataeff['PMTd'][i]})]
        mcabA = mcexprs[frozenset({dataeff['PMTa'][i], dataeff['PMTb'][i], dataeff['PMTA'][i]})]
        mcbcA = mcexprs[frozenset({dataeff['PMTb'][i], dataeff['PMTc'][i], dataeff['PMTA'][i]})]
        mcbdA = mcexprs[frozenset({dataeff['PMTb'][i], dataeff['PMTd'][i], dataeff['PMTA'][i]})]
    
        def eff(c3, c2, tag):
            e = c3 / c2
            return un.ufloat(e.n, np.sqrt(e.n * (1 - e.n) * 1/c2.n * (1 + 1/c2.n)), tag="dataeff_%deff%s" % (i, tag)) * e / e.n
        effAab = eff(countabA - noiseabA * time, countab - noiseab * time, 'Aab') * mcab / mcabA
        effAbc = eff(countbcA - noisebcA * time, countbc - noisebc * time, 'Abc') * mcbc / mcbcA
        effAbd = eff(countbdA - noisebdA * time, countbd - noisebd * time, 'Abd') * mcbd / mcbdA
    
        efficiencies.append((effAab, effAbc, effAbd))
   
    return fluxes2, fluxes2a, fluxes3, efficiencies

####### minimize #######

# p0 +/- dp0
up0 = [
    (2, 0.3), # total flux divided by 100
    (3.5, 1.5)
] + [(0.9, 0.1)] * len(dataeff['clock']) + [
    (0, 0.2)
]
p0 = [u[0] for u in up0]
simplex = [[u[0] - u[1] for u in up0]]
for i in range(len(up0)):
    simplex_element = [u[0] + u[1] for u in up0]
    simplex_element[i] = up0[i][0] - up0[i][1]
    simplex.append(simplex_element)

plt.ion()
fig = plt.figure('Simplex')
fig.clf()
ax = fig.add_subplot(111)
line, = ax.plot([p0[0]], [p0[1]], 'x', markersize=3)
lastline, = ax.plot([p0[0]], [p0[1]], 'kx', markersize=8, zorder=10)
a = [s[0] for s in simplex]
ax.set_xlim(min(a), max(a))
a = [s[1] for s in simplex]
ax.set_ylim(min(a), max(a))
fig.show()

def squares(parameters, args={}):
    total_flux = parameters[0] * 100
    distr_par = parameters[1]
    efficiencies = parameters[2:-1]
    lamda = 10 ** parameters[-1]
    
    # note: memoizing on distr_par would have very little effect
    fluxes2, fluxes2a, fluxes3, effs = f_fit(distr_par, options=args, lamda=lamda)
    fluxes = fluxes2 + fluxes2a + fluxes3

    vect = [flux - total_flux for flux in fluxes]
    for i in range(len(efficiencies)):
        vect += [eff - efficiencies[i] for eff in effs[i]]
    vect_nom = unp.nominal_values(vect)
    vect_cov = un.covariance_matrix(vect)
    
    # eigenvalues, transf = linalg.eigh(vect_cov)
    # orth_vect = transf.T.dot(vect_nom)
    # res = orth_vect / np.sqrt(eigenvalues)
    
    cov_inv = np.linalg.inv(vect_cov)
    Q = np.dot(vect_nom, np.dot(cov_inv, vect_nom))
    
    # save parameters, number of calls, value of function
    trace = args['trace']
    if not ('calls' in trace):
        trace['calls'] = 0
        trace['start'] = time.time()
        trace['pars'] = []
        for p in parameters:
            trace['pars'].append([])
        trace['Qs'] = []
    trace['calls'] += 1
    pars = trace['pars']
    for i in range(len(pars)):
        pars[i].append(parameters[i])
    trace['Qs'].append(Q)
    
    # log to the console
    if args.get('log', False):
        et = time.time() - trace['start']
        if trace['calls'] > 1:
            clear_lines(3, 70)
        print('squares: evaluation %d, elapsed time: %s' % (trace['calls'], '%.1f min' % (et / 60) if et >= 100 else '%.1f s' % et))
        print('parameters: %s' % (' '.join(['%.4f' % par for par in parameters])))
        print('sum of squares: %.4f' % Q)
    
    # plot flux, distr_par
    if args.get('plot', False):
        args['plotline'].set_xdata(pars[0])
        args['plotline'].set_ydata(pars[1])
        lastline.set_xdata(pars[0][-1:])
        lastline.set_ydata(pars[1][-1:])
        fig.canvas.draw()
        plt.pause(1e-17)
    
    return Q

# def curvature(trace, n=None, **kw):
#     m = len(trace['pars'])
#     if n is None:
#         n = int((m * (m + 1) / 2) * 2)
#     idxs = np.triu_indices(m)
#     x = np.array(trace['pars'])[:,-n:]
#     y = np.array(trace['Qs'])[-n:]
#     x0 = x[:,-1:]
#     Q0 = y[-1]
#     v = x - x0
#     V = np.empty((m, m))
#     def f(xs, *p):
#         V[idxs] = p
#         V.T[idxs] = p
#         Vinv = np.linalg.inv(V)
#         return 1/2 * np.einsum('ij,ik,kj->j', v, Vinv, v) + Q0
#     p0 = np.diag([0.01, 0.1] + [0.01] * (m-2))[idxs]**2
#     par, cov = optimize.curve_fit(f, x, y, p0=p0, **kw)
#     return par, cov
#     H = 1/2 * np.einsum('ik,jk->ijk', v, v)[idxs]
#     Vinv = np.einsum('ik,jk->ij', H, H)
#     V = np.linalg.inv(Vinv)
#     p = np.einsum('ij,jk,k->i', V, H, y)
#
#     covinv = np.empty((m, m))
#     covinv[idxs] = p
#     covinv.T[idxs] = p
#
#     return covinv

def plot_curvature(x0, ix, sx, n=10, geom={}):
    ixs = np.linspace(x0[ix] - sx, x0[ix] + sx, n)
    args = dict(log=True, plot=False, trace={}, geometry_factors=geom)
    x0 = np.copy(x0)
    squares(x0, args)
    for x in ixs:
        x0[ix] = x
        squares(x0, args)
    
    # plot
    trace = args['trace']
    x = np.array(trace['pars'][ix])
    y = np.array(trace['Qs'])
    fig = plt.figure('curvature_single')
    fig.clf()
    ax = fig.add_subplot(111)
    ax.plot(np.sort(x), y[np.argsort(x)], '-k.')
    fig.show()

def hessian(p0, dps, n='auto', geom={}, fig=None):
    """
    Parameters
    ----------
    p0: minimum
    dps: starting point to compute covariance is a box p0 +/- dps
    n: 'auto', or number of points used to compute a coefficient of the covariance matrix
    geom: empty dictionary, or geometry factors as computed by f_fit
    fig: None, or figure where fits used to compute the matrix are plotted
    
    Returns
    -------
    minima, Hessian (the inverse of the covariance) with uncertainties
    """
    assert(len(p0) == len(dps))
    
    # compute at p0 (eventually computing geometry uncertainty)
    args = dict(log=True, trace={}, geometry_factors=geom)
    squares(p0, args)
    Q0 = args['trace']['Qs'][-1]
    
    # fit function
    def parabola(x, a, x0, y0):
        return 1/2 * a * (x - x0)**2 + y0
    
    # plot
    if not (fig is None):
        fig.clf()
        G = gridspec.GridSpec(len(p0), len(p0))
    
    curvature = np.empty((len(p0), len(p0)), dtype=object)
    PAR = np.empty(len(p0), dtype=object)
    
    # compute diagonal elements
    for i in range(len(p0)):
        if n == 'auto':
            N = 20 if i == 1 else 4
        else:
            N = n
        # compute n points inside p0 +/- dps and find +3 chi^2 limits
        ps = np.linspace(p0[i] - dps[i], p0[i] + dps[i], N)
        vp = np.copy(p0)
        for p in ps:
            vp[i] = p
            squares(vp, args)
        x = np.concatenate((args['trace']['pars'][i][-N:], [p0[i]]))
        y = np.concatenate((args['trace']['Qs'][-N:], [Q0]))
        s = np.argsort(x)
        x = x[s]
        y = y[s]
        f = interpolate.interp1d(x, y)
        deltachi2 = 15 if i == 1 else 3
        L = optimize.bisect(lambda x: f(x) - (Q0 + deltachi2), x[0], p0[i])
        R = optimize.bisect(lambda x: f(x) - (Q0 + deltachi2), p0[i], x[-1])
        
        # compute n points inside +3 chi^2 limits and fit parabola
        ps = np.linspace(L, R, N)
        for p in ps:
            vp[i] = p
            squares(vp, args)
        x = np.array(args['trace']['pars'][i][-N:])
        y = np.array(args['trace']['Qs'][-N:]) / 2
        par, cov = optimize.curve_fit(parabola, x, y, p0=(1/((R-L)/6)**2, p0[i], Q0/2))
        assert(par[0] > 0)
        curvature[i, i] = un.ufloat(par[0], np.sqrt(cov[0,0]), tag='curv_%d%d' % (i, i))
        PAR[i] = un.ufloat(par[1], np.sqrt(cov[1,1]), tag='x_0_%d%d' % (i, i))
        
        # plot
        if not (fig is None):
            ax = fig.add_subplot(G[i,i])
            ax.plot(x, y, '.k', label="p%d" % i)
            xp = np.linspace(min(x), max(x), 100)
            ax.plot(xp, parabola(xp, *par), '-r', label='fit')
            ax.legend(fontsize='small')
    
    # compute off-diagonal elements
    for i in range(len(p0)):
        for j in range(i + 1, len(p0)):
            if n == 'auto':
                N = 10 if (i == 1 or j == 1) else 4
            else:
                N = n
        
            # compute N points along i, j scaled by 1 / sqrt(curvature)
            # conti sul quaderno alla data 2018-02-07
            cii = curvature[i,i]
            cjj = curvature[j,j]
            sigma_i = 1 / np.sqrt(cii.n)
            sigma_j = 1 / np.sqrt(cjj.n)
            ps_i = np.linspace(p0[i] - 3 * sigma_i, p0[i] + 3 * sigma_i, N)
            ps_j = np.linspace(p0[j] - 3 * sigma_j, p0[j] + 3 * sigma_j, N)
            vp = np.copy(p0)
            for k in range(len(ps_i)):
                vp[i] = ps_i[k]
                vp[j] = ps_j[k]
                squares(vp, args)
            x_i = np.array(args['trace']['pars'][i][-N:])
            x_j = np.array(args['trace']['pars'][j][-N:])
            x = x_i / sigma_i + x_j / sigma_j
            y = np.array(args['trace']['Qs'][-N:]) / 2
            fit_p0 = [
                1, # corresponds to cij = 0
                p0[i] / sigma_i + p0[j] / sigma_j,
                Q0 / 2
            ]
            par, cov = optimize.curve_fit(parabola, x, y, p0=fit_p0)
            ctpp = un.ufloat(par[0], np.sqrt(cov[0,0]), tag='curv_%d+%d' % (i, j))
            curvature[i, j] = (ctpp - 1) * umath.sqrt(cii * cjj)
            curvature[j, i] = curvature[i, j]
        
            # plot
            if not (fig is None):
                ax = fig.add_subplot(G[i,j])
                ax.plot(x, y, '.k', label="p%d+p%d" % (i,j))
                xp = np.linspace(min(x), max(x), 100)
                ax.plot(xp, parabola(xp, *par), '-r', label='fit')
                ax.legend(fontsize='small')
    
    return PAR, curvature

def res_plot(par, geom, cov=None, p1=None, **kw):
    par = np.copy(par)
    if not (p1 is None):
        par[1] = p1
    kwargs = dict(options={'geometry_factors': geom}, lamda=10**par[-1])
    kwargs.update(kw)
    fluxes2, fluxes2a, fluxes3, effs = f_fit(par[1], **kwargs)
    
    fig = plt.figure('residuals')
    fig.clf()
    G = gridspec.GridSpec(len(par) - 3, 3)
    
    # fluxes plot
    idxs = np.arange(len(fluxes2) + len(fluxes2a) + len(fluxes3))[::-1]
    ax = fig.add_subplot(G[:,:-1])
    ax.set_title('Flussi')
    kwargs = dict(fmt='.', capsize=4)
    ax.errorbar(unp.nominal_values(fluxes2), idxs[:len(fluxes2)], xerr=unp.std_devs(fluxes2), color='black', label='flussi "2"', **kwargs)
    ax.errorbar(unp.nominal_values(fluxes2a), idxs[len(fluxes2):len(fluxes2)+len(fluxes2a)], xerr=unp.std_devs(fluxes2a), color='lightgray', label='flussi "2a"', **kwargs)
    if len(fluxes3) > 0:
        ax.errorbar(unp.nominal_values(fluxes3), idxs[-len(fluxes3):], xerr=unp.std_devs(fluxes3), color='darkgray', label='flussi "3"', **kwargs)
    ax.set_yticks(idxs)
    ax.set_yticklabels([
        '%d&%d' % (data2['PMTA'][i], data2['PMTB'][i]) for i in range(len(fluxes2))
    ] + [
        '%d&%d' % (data2a['PMTA'][i], data2a['PMTB'][i]) for i in range(len(fluxes2a))
    ] + [
        '%d&%d&%d' % (data3['PMTA'][i], data3['PMTB'][i], data3['PMTC'][i]) for i in range(len(fluxes3))
    ])
    lims = ax.get_ylim()
    ax.plot([par[0] * 100] * 2, lims, '-k', label='fit')
    ax.set_ylim(lims)
    ax.legend(loc=0, fontsize='small')
    
    # effs plot
    idxs = np.arange(3)[::-1]
    for i in range(len(effs)):
        ax = fig.add_subplot(G[i,-1:])
        if i == 0:
            ax.set_title('Efficienze')
        ax.errorbar(unp.nominal_values(effs[i]), idxs, xerr=unp.std_devs(effs[i]), fmt='.k', capsize=4)
        ax.set_yticks(idxs)
        ax.set_yticklabels([
            '%d&%d' % (dataeff['PMTa'][i], dataeff['PMTb'][i]),
            '%d&%d' % (dataeff['PMTb'][i], dataeff['PMTc'][i]),
            '%d&%d' % (dataeff['PMTb'][i], dataeff['PMTd'][i])
        ])
        ax.set_ylim((-.5, 2.5))
        lims = ax.get_ylim()
        ax.plot([par[2+i]] * 2, lims, '-k')
        ax.set_ylim(lims)
    
    fig.show()

fit_options = dict(disp=True, xatol=2e-4, fatol=2e-3, initial_simplex=simplex)

print('computing geometrical uncertainty...')
args = dict(log=True, plot=True, plotline=line, trace={})
f_fit(p0[1], args, lamda=p0[-1])

print('first (of 3) fit...')
out1 = optimize.minimize(squares, p0, args=(args,), method='Nelder-Mead', options=fit_options)
trace1 = args['trace']
geom1 = args['geometry_factors']

print('recomputing geometrical uncertainty...')
args.pop('geometry_factors')
f_fit(out1.x[1], args, lamda=out1.x[-1])

line, = ax.plot([p0[0]], [p0[1]], 'x', markersize=3)
args['plotline'] = line
args['trace'] = {}

print('second (of 3) fit...')
out2 = optimize.minimize(squares, p0, args=(args,), method='Nelder-Mead', options=fit_options)
trace2 = args['trace']
geom2 = args['geometry_factors']

print('recomputing geometrical uncertainty...')
args.pop('geometry_factors')
f_fit(out2.x[1], args, lamda=out2.x[-1])

line, = ax.plot([p0[0]], [p0[1]], 'x', markersize=2)
args['plotline'] = line
args['trace'] = {}

print('third fit...')
out3 = optimize.minimize(squares, p0, args=(args,), method='Nelder-Mead', options=fit_options)
trace3 = args['trace']
geom3 = args['geometry_factors']

print('computing covariance...')
figcurv = plt.figure('curvature')
upar, H = hessian(out3.x, [0.1, 0.4] + [0.05] * len(dataeff['clock']) + [0.3], geom=geom3, fig=figcurv)
cov = np.linalg.inv(unp.nominal_values(H))
par = unp.nominal_values(upar)

print('upar:')
print(upar)
print('result:')
print(lab.format_par_cov(par, cov))

dof = len(data2['clock']) + len(data2a['clock']) + len(data3['clock']) + 3 * len(dataeff['clock']) - len(out3.x)
chisq = trace3['Qs'][-1]

print('chi2 = %.1f (dof = %d), chi2/dof = %.1f' % (chisq, dof, chisq / dof))

res_plot(out3.x, geom3, cov=cov)
