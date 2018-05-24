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
files=glob.glob('../daq/0508_na_forte_*cm.txt')

# vettori utili
d=array([])


# funzione di supporto
def p2(count,edges):

    X=edges[1:]-edges[:1]/2
    
    for j in range(2):
        if j==0:
            dom=X[X<504]
            cont=count[X<500]
        else:
            dom=X[X>496]
            cont=count[X>500]
             
        argmax=py.argmax(cont)
        cut = (dom[argmax]-40,dom[argmax]+40)
        ordinata=gvar.gvar(cont,py.sqrt(cont))
            
        if j==0:
            outdict,indict = fit_peak(dom,ordinata,bkg='exp',npeaks=1,cut=cut)
            beta=outdict['peak1_mean']
            sbeta=outdict['peak1_sigma']
        else:
            outdict,indict = fit_peak(dom,ordinata,bkg='exp',npeaks=1,cut=cut)
            neon=outdict['peak1_mean']
            sneon=outdict['peak1_sigma']
    
    return beta,sbeta,neon,sneon

for i in range len(files):
    files[i]=files.replace("\\","/")
    ch1,ch2,ch3,tr1,tr2,tr3,c2,c3,ts = lab4.loadtxt(files[i], unpack=True, usecols=(0,1,2,4,5,6,8,9,12))
    d=append(d,scanf.scanf('../daq/0508_na_forte_%dcm.txt',files[i]))
    
    