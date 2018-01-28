import numpy as np
import montecarlo as mc
import uncertainties as un

####### load data #######

def loadtxtlbs(filename, labels, prefit=True):
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

data2lbs = ['clock', 'A', 'B', 'A&B', 'a&c', 'a&b&c', 'a&b&c&A', 'a&b&c&B', 'PMTA', 'PMTB', 'PMTa', 'PMTb', 'PMTc', 'prefit']
data2albs = ['clock', 'A', 'B', 'A&B', 'a&b', 'a&b&B', 'a&b&A', 'a&b&A&B', 'PMTA', 'PMTB', 'PMTa', 'PMTb', 'prefit']
data3lbs = ['clock', 'C', 'B', 'A&B&C', 'a&b', 'a&b&C', 'a&b&B', 'a&b&A', 'PMTA', 'PMTB', 'PMTC', 'pmta', 'pmtb', 'prefit']
dataefflbs = ['clock', 'a&b', 'a&b&A', 'b&c', 'b&c&A', 'b&d', 'b&d&A', 'PMTA', 'PMTa', 'PMTb', 'PMTc', 'PMTd', 'prefit']

data2 = loadtxtlbs('fitdata2.txt', data2lbs)
data2a = loadtxtlbs('fitdata2a.txt', data2albs)
data3 = loadtxtlbs('fitdata3.txt', data3lbs)
dataeff = loadtxtlbs('fitdataeff.txt', dataefflbs)

data2sigA = un.ufloat(38, 2) * 1e-9
data2sigB = un.ufloat(37, 2) * 1e-9

####### prepare monte carlo #######

# draw samples
mcobj = mc.MC(*[mc.pmt(i+1) for i in range(6)])
mcobj.random_samples(N=100000)

mcgeom = mc.MC(*[mc.pmt(i+1) for i in range(6)])
mcgeom.random_samples(N=1000)

# create list of expressions to compute
mclist = []
for i in range(len(data2['clock'])):
    mclist += [
        {data2['PMTA'][i], data2['PMTB'][i]}
    ]
for i in range(len(data2a['clock'])):
    mclist += [
        {data2a['PMTA'][i], data2a['PMTB'][i]},
        {data2a['PMTa'][i], data2a['PMTb'][i]},
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
    counts = mcobj.count(*cexprs[pivot])
    # save
    expr = exprs[pivot]
    for i in range(len(expr)):
        mcexprs[expr[i]] = counts[i] / mcobj.number_of_rays * mcobj.pivot_horizontal_area

# compute geometrical uncertainty
mcsegeom = dict()
for expr in mcexprs.keys():
    mcsegeom[expr] = []
# compute acceptances for each geometry sample
for j in range(1000):
    for pivot in exprs.keys():
        mcgeom.run(pivot_scint=pivot, randgeom=True)
        counts = mcgeom.count(*cexprs[pivot])
        expr = exprs[pivot]
        for i in range(len(expr)):
            mcsegeom[expr[i]].append(counts[i].n / mcgeom.number_of_rays * mcgeom.pivot_horizontal_area.n)
# compute standard deviation due to geometry
for expr in mcsegeom.keys():
    samples = mcsegeom[expr]
    sd = np.std(samples, ddof=1)
    val = mcexprs[expr]
    mcexprs[expr] = (val, un.ufloat(1, sd / val.n))

# process data2
fluxes = []
for i in range(len(data2['clock'])):
    time = un.ufloat(data2['clock'][i], 0.5) / 1000
    
    count = lambda label: un.ufloat(data2[label][i], np.sqrt(data2[label][i]))
    countA = count('A')
    countB = count('B')
    countAB = count('A&B')
    
    rateA = countA / time
    rateB = countB / time
    rateAB = countAB / time
    
    countabc = count('a&b&c')
    countabcA = count('a&b&c&A')
    countabcB = count('a&b&c&B')
    
    eff = lambda c3, c2: un.ufloat(c3.n / c2.n, c3.n/c2.n * (1 - c3.n/c2.n) * 1/c2.n * (1 + 1/c2.n))
    effA = eff(countabcA, countabc)
    effB = eff(countabcB, countabc)
    
    mcAB = mcexprs[frozenset({data2['PMTA'][i], data2['PMTB'][i]})]
    
    noiseAB = rateA * rateB * (data2sigA + data2sigB)
    
    flux = (rateAB - noiseAB) / (mcAB[0] * mcAB[1] * effA * effB)
    fluxes.append(flux)
    