# file copiato da Marasciulli/monte carlo dati.py e poi modificato
# modifiche: non fare il log di 0
# non mettere troppi bin nell'istogramma

from pylab import *
import pylab as py
from scipy.special import loggamma
import lab
from matplotlib.gridspec import GridSpec

# dati
n=py.loadtxt("pm2 10s.txt",unpack=True)

mu=py.mean(n)
N=10**4 # N del monte carlo

def poisson(k,mu):
    return py.exp( k*log(mu)-mu-loggamma(1+k)   ) if mu != 0 else where(k != 0, 0, 1)   
    
def chi2(ki):
    k,c=py.unique(ki,return_counts=True)
    e=len(ki)*poisson(k,py.mean(ki))
    return py.sum((c-e)**2/py.where(e!=0,e,1))

chi2s=py.empty(N)
for i in range(N):
    cont=py.poisson(mu,len(n))
    chi2s[i]=chi2(cont)
    
pvalue = py.sum(chi2s > chi2(n)) / N

figure('dati', figsize=[ 6.86,  2.81]).set_tight_layout(True)
clf()
grid = GridSpec(1, 3)

subplot(grid[0,:2])

ks = arange(min(n), max(n) + 1)
bar(ks, poisson(ks, mu) * len(n), label='Poissoniana, $\\mu=\\langle k \\rangle$', color='lightgray', width=1)

hist(n, bins=arange(min(n)-0.5, max(n)+1.5)+.2, label='Dati, $\\langle k \\rangle=$%s' % (lab.util_format(mu, sqrt(mu)/sqrt(len(n)), pm='@', comexp=False).split(' @')[0],), color='darkgray', width=.6)

legend(loc=1, fontsize='small')
xlabel('Conteggio')
ylabel('Occorrenze')
title('Test di conteggio')

subplot(grid[0,-1])

hist(log(1+chi2s), bins='auto', color='darkgray', label='MC N=%d' % N)
lims = ylim()
plot([log(1+chi2(n))]*2, lims, '-k', label='$\\log(1+\\chi^2(\\mathrm{dati}))$\n$p=%.2f$' % pvalue)
ylim(lims)

ylabel('Occorrenze')
xlabel('$\\log(1+\\chi^2)$')
legend(loc=1, fontsize='small')
title('MC $\\chi^2$')

py.show()
