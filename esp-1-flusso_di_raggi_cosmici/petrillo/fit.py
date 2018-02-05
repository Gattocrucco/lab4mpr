import numpy as np
import montecarlo as mc
import uncertainties as un
from uncertainties import unumpy as unp
import copy
from collections import OrderedDict
from scipy import optimize, linalg
import time
from matplotlib import pyplot as plt

####### diagnostics #######

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

####### load data #######

def loadtxtlbs(filename, labels, prefit=True):
    data = np.loadtxt(filename, unpack=True)
    datadict = dict()
    if len(data) != len(labels):
        raise RuntimeError('length of data %d != length of labels %d' % (len(data), len(labels)))
    for i in range(len(data)):
        datadict[labels[i]] = data[i]
    p = datadict['prefit']
    c = p == 1
    for k in datadict.keys():
        if k != 'prefit':
            datadict[k] = datadict[k][c]
    datadict['prefit'] = p[c]
    return datadict

data2lbs = ['clock', 'A', 'B', 'A&B', 'a&c', 'a&b&c', 'a&b&c&A', 'a&b&c&B', 'PMTA', 'PMTB', 'PMTa', 'PMTb', 'PMTc', 'prefit']
data2albs = ['clock', 'A', 'B', 'A&B', 'a&b', 'a&b&B', 'a&b&A', 'a&b&A&B', 'PMTA', 'PMTB', 'PMTa', 'PMTb', 'prefit']
data3lbs = ['clock', 'C', 'B', 'A&B&C', 'a&b', 'a&b&C', 'a&b&B', 'a&b&A', 'PMTA', 'PMTB', 'PMTC', 'pmta', 'pmtb', 'prefit']
dataefflbs = ['clock', 'a&b', 'a&b&A', 'b&c', 'b&c&A', 'b&d', 'b&d&A', 'PMTA', 'PMTa', 'PMTb', 'PMTc', 'PMTd', 'prefit']

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
mcobj = mc.MC(*[mc.pmt(i+1) for i in range(6)])
mcobj.random_samples(N=100000)

mcgeom = mc.MC(*[mc.pmt(i+1) for i in range(6)])
mcgeom.random_samples(N=316)
mcgeom.sample_geometry(316)

# create list of expressions to compute
mclist = []
for i in range(len(data2['clock'])):
    mclist += [
        {data2['PMTA'][i], data2['PMTB'][i]}
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
        {data3['PMTA'][i], data3['PMTB'][i], data3['PMTC'][i]}
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

def f_fit(distr_par):
    # compute rays
    distr = lambda x: x ** (1 / (1 + distr_par)) # vedi logbook:fit
    mcobj.ray(distr)
    mcgeom.ray(distr)

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
        mcexprs[mcsegeom_keys[i]] *= geom_factors[i]

    # process data2
    fluxes2 = []
    for i in range(len(data2['clock'])):
        assert(data2['prefit'][i])
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
    
        eff = lambda c3, c2, tag: un.ufloat(c3.n / c2.n, np.sqrt(c3.n/c2.n * (1 - c3.n/c2.n) * 1/c2.n * (1 + 1/c2.n)), tag="data2_%deff%s" % (i, tag))
        effA = eff(countabcA, countabc, 'A')
        effB = eff(countabcB, countabc, 'B')
    
        mcAB = mcexprs[frozenset({data2['PMTA'][i], data2['PMTB'][i]})]
    
        noiseAB = rateA * rateB * (data2sigA + data2sigB)
    
        flux = (rateAB - noiseAB) / (mcAB * effA * effB)
        fluxes2.append(flux)

    # process data2a
    fluxes2a = []
    for i in range(len(data2a['clock'])):
        assert(data2a['prefit'][i])
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
    
        noiseAB = rateA * rateB * (data2asigA + data2asigB)
    
        flux = (rateAB - noiseAB) / (mcAB * effA * effB)
        fluxes2a.append(flux)

    # process data3
    fluxes3 = []
    for i in range(len(data3['clock'])):
        assert(data3['prefit'][i])
        time = un.ufloat(data3['clock'][i], 0.5, tag="data3_%dtime" % i) * 1e-3
    
        count = lambda label: un.ufloat(data3[label][i], np.sqrt(data3[label][i]), tag="data3_%dcount%s" % (i, label))
        countABC = count('A&B&C')
        countab = count('a&b')
        countabA = count('a&b&A')
        countabB = count('a&b&B')
        countabC = count('a&b&C')
    
        rateABC = countABC / time    
        noiseab = data3ratea * data3rateb * (data3siga + data3sigb)
    
        def eff(c3, c2, tag):
            e = c3.n / c2
            return un.ufloat(e.n, np.sqrt(e.n * (1 - e.n) * 1/c2.n * (1 + 1/c2.n)), tag="data3_%deff%s" % (i, tag)) * e / e.n
        effA = eff(countabA, countab.n - noiseab * time, 'A')
        effB = eff(countabB, countab.n - noiseab * time, 'B')
        effC = eff(countabC, countab.n - noiseab * time, 'C')
    
        mcABC = mcexprs[frozenset({data3['PMTA'][i], data3['PMTB'][i]}, data3['PMTC'][i])]
    
        flux = rateABC / (mcABC * effA * effB * effC)
        fluxes3.append(flux)

    dataefflbs = ['clock', 'a&b', 'a&b&A', 'b&c', 'b&c&A', 'b&d', 'b&d&A', 'PMTA', 'PMTa', 'PMTb', 'PMTc', 'PMTd', 'prefit']
    # process dataeff
    efficiencies = []
    for i in range(len(dataeff['clock'])):
        assert(dataeff['prefit'][i])
        time = un.ufloat(dataeff['clock'][i], 0.5, tag="dataeff_%dtime" % i) * 1e-3
    
        count = lambda label: dataeff[label][i]
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
    
        noiseab = ratea * rateb * (dataeffsiga + dataeffsigb) * time
        noisebc = rateb * ratec * (dataeffsigb + dataeffsigc) * time
        noisebd = rateb * rated * (dataeffsigb + dataeffsigd) * time
    
        mcab = mcexprs[frozenset({dataeff['PMTa'][i], dataeff['PMTb'][i]})]
        mcbc = mcexprs[frozenset({dataeff['PMTb'][i], dataeff['PMTc'][i]})]
        mcbd = mcexprs[frozenset({dataeff['PMTb'][i], dataeff['PMTd'][i]})]
        mcabA = mcexprs[frozenset({dataeff['PMTa'][i], dataeff['PMTb'][i], dataeff['PMTA'][i]})]
        mcbcA = mcexprs[frozenset({dataeff['PMTb'][i], dataeff['PMTc'][i], dataeff['PMTA'][i]})]
        mcbdA = mcexprs[frozenset({dataeff['PMTb'][i], dataeff['PMTd'][i], dataeff['PMTA'][i]})]
    
        def eff(c3, c2, tag):
            e = c3 / c2
            return un.ufloat(e.n, np.sqrt(e.n * (1 - e.n) * 1/c2.n * (1 + 1/c2.n)), tag="dataeff_%deff%s" % (i, tag)) * e / e.n
        effAab = eff(countabA, countab - noiseab, 'Aab') * mcab / mcabA
        effAbc = eff(countbcA, countbc - noisebc, 'Abc') * mcbc / mcbcA
        effAbd = eff(countbdA, countbd - noisebd, 'Abd') * mcbd / mcbdA
    
        efficiencies.append((effAab, effAbc, effAbd))
   
    return fluxes2 + fluxes2a + fluxes3, efficiencies

####### minimize #######

p0 = [250, 5] + [0.90] * len(dataeff['clock'])
simplex = np.array([
    [p0[0]]
])
calls = 0
start = time.time()
plt.ion()
fig = plt.figure('Simplex')
fig.clf()
ax = fig.add_subplot(111)
line, = ax.plot([p0[0]], [p0[1]], '-xr')
fig.show()
pars = []
for p in p0:
    pars.append([])
Qs = []
def squares(parameters):
    total_flux = parameters[0]
    distr_par = parameters[1]
    efficiencies = parameters[2:]

    fluxes, effs = f_fit(distr_par)

    vect = [flux - total_flux for flux in fluxes]
    for i in range(len(efficiencies)):
        vect += [eff - efficiencies[i] for eff in effs[i]]
    vect_nom = unp.nominal_values(vect)
    vect_cov = un.covariance_matrix(vect)
    
    # eigenvalues, transf = linalg.eigh(vect_cov)
    # orth_vect = transf.T.dot(vect_nom)
    # res = orth_vect / np.sqrt(eigenvalues)
    
    inverse = np.linalg.inv(vect_cov)
    Q = np.dot(vect_nom, np.dot(inverse, vect_nom))
    
    global calls
    calls += 1
    et = time.time() - start
    print('------------------------------')
    print('squares: evaluation %d, elapsed time: %s' % (calls, '%.1f min' % (et / 60) if et >= 100 else '%.1f s' % et))
    print('parameters: %s' % (' '.join(['%.3f' % par for par in parameters])))
    print('sum of squares: %.3f' % Q)
    for i in range(len(pars)):
        pars[i].append(parameters[i])
    Qs.append(Q)
    line.set_xdata(pars[0])
    line.set_ydata(pars[1])
    fig.canvas.draw()
    plt.pause(1e-17)
    time.sleep(0.05)
        
    return Q

# bounds = [(100, 400), (1, 10)] + [(0, 1)] * len(dataeff['clock'])
# scale = [10, 1] + [0.03] * len(dataeff['clock'])
# result = optimize.least_squares(squares, p0, diff_step=1e-5, verbose=2)
# out = optimize.minimize(squares, p0, method='Nelder-Mead', options=dict(disp=True, xatol=1e-4, fatol=1e-3))

