## FOTONI POTENTI

from pylab import *
import sys
import lab4
import lab
from fit_peak_originale import fit_peak
import gvar
import glob
import scanf

# importo i file
files=glob.glob('../daq/0508_naforte_*cm.txt')

# vettori utili
d=array([])
beta=empty(len(files),dtype=object)
sbeta=empty(len(files),dtype=object)
neon=empty(len(files),dtype=object)
sneon=empty(len(files),dtype=object)

# funzione di supporto
def p2(count,edges):
    import pylab as py

    X=(edges[:-1]+edges[1:])/2
    
    for j in range(2):
        if j==0:
            dom=X[X<500]
            cont=count[X<500]
        else:
            dom=X[X>500]
            cont=count[X>500]
        print('d=',d,'j=',j)
        argmax=py.argmax(cont)
        cut = (dom[argmax]-30,dom[argmax]+30)
        ordinata=gvar.gvar(cont,py.sqrt(cont))
            
        if j==0:
            outdict,indict = fit_peak(edges[X>500],ordinata,bkg='exp',npeaks=1,cut=cut,ax=prova)
            beta=outdict['peak1_mean']
            sbeta=outdict['peak1_sigma']
        else:
            outdict,indict = fit_peak(dom,ordinata,bkg='exp',npeaks=1,cut=cut,ax=prova)
            neon=outdict['peak1_mean']
            sneon=outdict['peak1_sigma']
    
    return beta,sbeta,neon,sneon

for i in range(len(files)):
    files[i]=files[i].replace("\\","/")
    ch1,ch2,ch3,tr1,tr2,tr3,c2,c3,ts = lab4.loadtxt(files[i], unpack=True, usecols=(0,1,2,4,5,6,8,9,12))
    
    d=append(d,scanf.scanf('../daq/0508_naforte_%dcm.txt',files[i]))
    
    hu=figure('%s'%files[i])
    prova=hu.add_subplot(111)
    ch1=ch1[(ch1>250) & (ch2<100) & (ch1<1000)]
    con,bor,_=hist(ch1,bins=arange(0,1200//8)*8,histtype='step')
    beta[i],sbeta[i],neon[i],sneon[i]=p2(con,bor)
    
figure('').set_tght_layout(True)
clf()
rc('font',size=14)

subplot(211)
title("Sorgente con attività elevata")
grid(linestyle=':')
minorticks_on()

ylabel('valore ADC  [digit]')
errorbar(d,gvar.mean(beta),gvar.sdev(beta),fmt='.r',capsize=2,markersize=3,label='annichilazione')
errorbar(d,gvar.mean(neon),gvar.sdev(neon),fmt='.b',capsize=2,markersize=3,label='neon')

subplot(212)
grid(linestyle=':')
minorticks_on()

xlabel('distanza  [cm]')
errorbar(1,1,1,1,fmt='.k',capsize=2,markersize=3)

legend()
show()