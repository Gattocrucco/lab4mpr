# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import sys
import lab
from scipy.special import chdtrc

filenames = sys.argv[1:]

fig = plt.figure('histo')
fig.clf()
fig.set_tight_layout(True)

def unroll_time(t):
    # non è ancora del tutto buona perché non tiene conto del fatto
    # che ci sono più dati a tempo fissato, però per i nostri scopi
    # va bene.
    tmax = 6553.5
    # preso da max(t)
    # bisogna sommare 65535 e non 65536 perché min(t) == 0.1
    diff = np.diff(t)
    cycles = np.concatenate([[0], np.cumsum(diff < 0)])
    return t + tmax * cycles

if len(filenames) > 1:
    datasets = []
    for filename in filenames:
        print('loading %s...' % (filename,))
        t, ch1, ch2 = np.loadtxt(filename, unpack=True)
        datasets.append(ch1)
    ax = fig.add_subplot(111)
    nbinspow = min(int(np.ceil(np.log2(np.sqrt(max([len(ds) for ds in datasets]))))), 12)
    edges = np.arange(2 ** 12 + 1)[::2 ** (12 - nbinspow)] - 0.5
    ax.hist(datasets, bins=edges, density=True, histtype='step', label=filenames)
    ax.legend(loc='upper right', fontsize='small')
    ax.set_xlabel('canale ADC')
    ax.set_ylabel('densita')

## MODIFICATO PER FITTARE GAUSSIANE 

elif len(filenames) == 1:
    filename = filenames[0]
    print('loading %s...' % (filename,))
    rolled_t, ch1, ch2 = np.loadtxt(filename, unpack=True)
    t = unroll_time(rolled_t)
    ax1 = fig.add_subplot(211)
    nbinspow = min(int(np.ceil(np.log2(np.sqrt(len(ch1))))), 12)
    edges = np.arange(2 ** 12 + 1)[::2 ** (12 - nbinspow)] - 0.5
    cont,_,_=ax1.hist(ch1, bins=edges, histtype='step', label=filename)
    ax1.legend(loc='best', fontsize='small')
    ax1.set_ylabel('conteggio')
    ax1.set_xlabel('canale ADC')
    ax2 = fig.add_subplot(212)
    ax2.plot(t, ch1, '.', markersize=2)
    ax2.set_xlabel('tempo')
    ax2.set_ylabel('canale ADC')

else:
    print('no filenames specified.')

fit=True  #filename= 0316ang0.dat
if fit==True:
    
    def gaussiana(x,N,u,sigma):
        return (N/(sigma*np.sqrt(2*np.pi)))*np.exp(-((x-u)**2)/(2*sigma**2))
    
    taglio=[2900,3350] # a mano
    edges=edges[:-1]
    X=np.array([])
    Y=np.array([])
    for i in range(len(edges)):
        if edges[i]>=taglio[0]:
            X=np.append(X,edges[i])
            Y=np.append(Y,cont[i])
    
    val=[10**4,3200,100]
    out=lab.fit_curve(gaussiana,X,Y,p0=val,dy=np.sqrt(Y),print_info=1,absolute_sigma=True,method="leastsq")
    
    z=np.linspace(taglio[0],taglio[1],1000)
    ax1.plot(z,gaussiana(z,*out.par),color="red")

fig.show()

print("")
dof=len(X)-len(out.par)
p=chdtrc(dof,out.chisq)
print("chi quadro=",out.chisq,"+-",np.sqrt(2*dof),"  dof=",dof)
print("p valore=",p)