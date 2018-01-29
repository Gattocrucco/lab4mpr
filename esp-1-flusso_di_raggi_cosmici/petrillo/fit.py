import numpy as np
import montecarlo as mc
import uncertainties as un
from uncertainties import unumpy as unp
import copy

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
mcgeom.random_samples(N=500)

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

# def f_fit(total_flux, distr_par):
total_flux=0
distr_par=2
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
mcsegeom = dict()
for expr in mcexprs.keys():
    mcsegeom[expr] = []
# compute acceptances for each geometry sample
for j in range(100):
    randgeom = True
    for pivot in exprs.keys():
        mcgeom.run(pivot_scint=pivot, randgeom=randgeom)
        randgeom = 'last'
        counts = mcgeom.count(*cexprs[pivot])
        expr = exprs[pivot]
        for i in range(len(expr)):
            mcsegeom[expr[i]].append(counts[i].n / mcgeom.number_of_rays * mcgeom.pivot_horizontal_area)
# compute uncertainty due to geometry
mcsegeom_keys = list(mcsegeom.keys())
mcsegeom_array = np.array([mcsegeom[key] for key in mcsegeom_keys])
mcsegeom_cov = np.cov(mcsegeom_array)
assert(len(mcsegeom_cov) == len(mcsegeom_keys))
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

def errorsummary(x):
    from collections import OrderedDict
    comps = x.error_components()
    
    varmc = 0
    vargeom = 0
    varother = 0
    tags = []
    sds = []
    for (var, sd) in comps.items():
        if var.tag == 'mc':
            varmc += sd ** 2
        elif var.tag == 'geom':
            vargeom += sd ** 2
        elif var.tag is None:
            varother += sd ** 2
        else:
            tags.append(var.tag)
            sds.append(sd)
    
    if varmc > 0:
        tags.append('mc')
        sds.append(np.sqrt(varmc))
    if vargeom > 0:
        tags.append('geom')
        sds.append(np.sqrt(vargeom))
    if varother > 0:
        tags.append('_other')
        sds.append(np.sqrt(varother))
    
    idx = np.argsort(sds)[::-1]
    d = OrderedDict()
    for i in idx:
        d[tags[i]] = sds[i]
    
    return d
