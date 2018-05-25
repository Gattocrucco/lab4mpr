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
def p2(count,edges,dd):
    
    for j in range(2):
        if j==0:
            max_i=argmax(count[edges[1:]<600])
            ex=edges[edges<600][max_i]
        else:
            max_i=argmax(count[edges[1:]>600])
            ex=edges[edges>600][max_i]
            
        cut=(ex-30,ex+30)
        ordinata=gvar.gvar(count,sqrt(count))
            
        if j==0:
            outdict,indict = fit_peak(edges,ordinata,bkg='exp',npeaks=1,cut=cut,print_info=0)
            beta=outdict['peak1_mean']
            sbeta=outdict['peak1_sigma']
        elif j==1 and ((dd!=3) and (dd!=4.5) and (dd!=46)):
            outdict,indict = fit_peak(edges,ordinata,bkg=None,npeaks=1,cut=cut,print_info=0)
            neon=outdict['peak1_mean']
            sneon=outdict['peak1_sigma']
        else:
            neon=nan
            sneon=nan
    
    return beta,sbeta,neon,sneon

for i in range(len(files)):
    files[i]=files[i].replace("\\","/")
    ch1,ch2,ch3,tr1,tr2,tr3,c2,c3,ts = lab4.loadtxt(files[i], unpack=True, usecols=(0,1,2,4,5,6,8,9,12))
    
    d=append(d,scanf.scanf('../daq/0508_naforte_%fcm.txt',files[i]))
    
    ch1=ch1[(ch1>250) & (ch2<100) & (ch1<1000)]
    con,bor=histogram(ch1,bins=arange(0,1200//8)*8)
    beta[i],sbeta[i],neon[i],sneon[i]=p2(con,bor,d[i])
    
figure('andre').set_tight_layout(True)
clf()
rc('font',size=14)

ax=subplot(312)

grid(linestyle=':')
minorticks_on()

ylabel('valore ADC')
errorbar(d,gvar.mean(beta),gvar.sdev(beta),fmt='.r',capsize=2,label='annichilazione')
legend()
xscale('log')

subplot(311,sharex=ax)
title("Sorgente con attività elevata")

grid(linestyle=':')
minorticks_on()
ylabel('[digit]')
errorbar(d,gvar.mean(neon),gvar.sdev(neon),fmt='.b',capsize=2,label='neon')
legend()
xscale('log')

subplot(313,sharex=ax)
grid(linestyle=':',which='both')
minorticks_on()

xlabel('distanza  [cm]')

dist,evt,ms=lab4.loadtxt('../dati/0508_dist.txt',unpack=True)
evt=gvar.gvar(evt,sqrt(evt))
t=ms/1000
rate=evt/t
errorbar(dist,gvar.mean(rate),gvar.sdev(rate),fmt='.k',capsize=2)

xscale('log')
yscale('log')
ylabel('rate  [1/s]')


# altra figura

fig=figure('Jack')
fig.set_tight_layout(True)
fig.clf()

neo, bet, rd = fig.subplots(3, 1, sharex=True)

# 311
neo.grid(linestyle=':')
neo.minorticks_on()

neo.set_ylabel('canale ADC')
neo.set_xscale('log')
neo.errorbar(gvar.mean(rate),gvar.mean(neon),xerr=gvar.sdev(rate),yerr=gvar.sdev(neon),fmt='.b',capsize=2,label='neon')


# 312
bet.grid(linestyle=':')
bet.minorticks_on()

bet.set_ylabel('canale ADC')
bet.set_xscale('log')
bet.errorbar(gvar.mean(rate),gvar.mean(beta),xerr=gvar.sdev(rate),yerr=gvar.sdev(beta),fmt='.r',capsize=2,label='annichilazione')


# 313
rd.grid(linestyle=':')
rd.minorticks_on()

rd.set_ylabel('distanza  [cm]')
rd.set_xlabel('rate  [1/s]')
rd.set_xscale('log')
rd.set_yscale('log')

rd.errorbar(gvar.mean(rate),dist,xerr=gvar.sdev(rate),fmt='.k',capsize=2)

neo.legend()
bet.legend()
show()