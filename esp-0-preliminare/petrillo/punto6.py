# file copiato da Marasciulli/monte carlo dati.py e poi modificato
# modifiche: non fare il log di 0
# non mettere troppi bin nell'istogramma

from pylab import *
import pylab as py
from scipy.special import loggamma

# dati
n=py.loadtxt("pm2 1s.txt",unpack=True)

mu=py.mean(n)
N=10**3 # N del monte carlo

def maxlogL(ki):
    ni=len(ki)
    km=py.mean(ki)
    return ni*km*log(km)-ni*km-py.sum(loggamma(1+ki)) if km != 0 else 0
    
def poisson(k,mu):
    return py.exp( k*log(mu)-mu-loggamma(1+k)   ) if mu != 0 else where(k != 0, 0, 1)   
    
def chi2(ki):
    k,c=py.unique(ki,return_counts=True)
    e=len(ki)*poisson(k,py.mean(ki))
    return py.sum((c-e)**2/py.where(e!=0,e,1))


like=py.empty((N,2))
for i in range(N):
    cont=py.poisson(mu,len(n))
    s=maxlogL(cont)
    like[i,0]=s
    like[i,1]=chi2(cont)
    
perc=py.empty(2)
perc[0]=py.sum(like[:,0]<maxlogL(n))
perc[1]=py.sum(like[:,1]>chi2(n))

perc /= N

figure('dati')
clf()
hist(n, bins=arange(min(n)-.5, max(n)+1.5, 1))

py.figure("max log L Poisson")
py.clf()

py.subplot(121)
py.hist(like[:,0],bins="sqrt" if N <= 10**4 else 100,color="blue",label="p = %.3f" % perc[0])
lims=py.ylim()
py.plot([maxlogL(n)]*2,lims,'-r')
py.legend()

py.subplot(122)
py.hist(like[:,1],bins="sqrt" if N <= 10**4 else 100,color="green",label="p = %.3f" % perc[1])
lims=py.ylim()
oriz=py.std(like[:,1])
py.xlim(0,oriz*3)
py.plot([chi2(n)]*2,lims,'-r')
py.legend()

py.show()