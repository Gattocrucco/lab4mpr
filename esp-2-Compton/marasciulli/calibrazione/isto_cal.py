## CALIBRAZIONE CON LE SORGENTI
import pylab as plt
import numpy as np
from pylab import *
from scipy.optimize import curve_fit
from scipy.special import chdtrc
import lab

cartella ="esp-2-Compton/dati/cal/"
data ="22feb-"
elements= [["cs","Cs 137", 0.662, [3700,4000]],
           ["am","Am 241", 0.059, [320,380]],
           ["na","Na 22",  0.511, [2900,3100]],
           ["na","Na 22",  1.2, [7100,7500]],
           ["co","Co 60", "1.17,1.33", [6450,7850]]]

#definition______________________________________
def bar_line(edges, counts, ax=None, **kwargs):
    dup_edges = np.empty(2 * len(edges))
    dup_edges[2 * np.arange(len(edges))] = edges
    dup_edges[2 * np.arange(len(edges)) + 1] = edges
    dup_counts = np.zeros(2 * len(edges))
    dup_counts[2 * np.arange(len(edges) - 1) + 1] = counts
    dup_counts[2 * np.arange(len(edges) - 1) + 2] = counts
    if ax is None:
        ax = plt.gca()
    return ax.plot(dup_edges, dup_counts, **kwargs)
    
def distr(x,N,u,sigma,m,q):
        return N*np.e**(-1*((x-u)**2)/(2*sigma**2)) + (m*x+q +abs(m*x+q))/2
    
def gaus2(x,N1,u1,sigma1,N2,u2,sigma2,m,q):
        return N1*np.e**(-1*((x-u1)**2)/(2*sigma1**2)) + N2*e**(-1*((x-u2)**2)/(2*sigma2**2)) + (m*x+q +abs(m*x+q))/2
        

#____________________________________________________

for a in elements[::]:
    nome=a[0]
    el=a[1]
    file=cartella+"histo-"+data+nome+".dat"
    energy = a[2]
    sin=a[3][0]
    dex=a[3][1]
    
    Y = loadtxt(file,unpack=True)
    X = arange(len(Y)+1)
    
    _Y = Y[sin:dex]
    _X = arange(dex-sin)+0.5+sin
    _dy=sqrt(_Y)
    
    # creazione istogramma   
    if(nome == "co"):
        val=[10**2,(3*sin+dex)/4,(dex-sin)/4,10**2,(sin+3*dex)/4,(dex-sin)/4,-0.1,0]
        popt,pcov=curve_fit(gaus2,_X,_Y,sigma=_dy,p0=val, maxfev=10000)
        # val = popt
        # sin = int(popt[1]-1.4*abs(popt[2]))
        # dex = int(popt[4]+1.8*abs(popt[5]))
        # _Y = Y[sin:dex]
        # _X = arange(dex-sin)+0.5+sin
        # _dy=sqrt(_Y)
        #popt,pcov=curve_fit(gaus2,_X,_Y,sigma=_dy,p0=val, maxfev=10000)
        
    else:
        val=[10**3,(sin+dex)/2,(dex-sin)/2,0,0]
        popt,pcov=curve_fit(distr,_X,_Y,sigma=_dy,p0=val, maxfev=10000)
        val = popt
        sin = int(popt[1]-1.8*abs(popt[2]))
        dex = int(popt[1]+1.8*abs(popt[2]))
        _Y = Y[sin:dex]
        _X = arange(dex-sin)+0.5+sin
        _dy=sqrt(_Y)
        popt,pcov=curve_fit(distr,_X,_Y,sigma=_dy,p0=val, maxfev=10000)
        
    
    #plot______________________________________________________
    
    #plot su tutto lo spettro
    figure(el+"_spectrum").set_tight_layout(True)
    
    rc("font",size=14)
    title("Calibrazione", size=16)
    grid(color="black",linestyle=":")
    minorticks_on()
    xlabel("energia [digit]")
    ylabel("conteggi")
    
    #bar(X*fis,Y,width=lbin*fis)
    if(energy!=elements[3][2]):
        bar_line(X, Y)
    
    z=linspace(sin,dex,1000)
    if(nome=="co"):
        plot(z,gaus2(z,*popt),color="red",linewidth=3)
    else:
        plot(z,distr(z,*popt),color="red",linewidth=3)
    
    #zoom sulla gaussiana    
    figure(el+"_"+str(energy)+"MeV").set_tight_layout(True)
    rc("font",size=14)
    title("Fit fotopicco a "+str(energy)+"MeV nello spettro "+el, size=16)
    grid(color="black",linestyle=":")
    minorticks_on()
    xlabel("energia [digit]")
    ylabel("conteggi")
    
    errorbar(_X,_Y, _dy, linestyle="", marker=".", color="black")
    if(nome=="co"):
        plot(z,gaus2(z,*popt),color="red",linewidth=4)
    else:
        plot(z,distr(z,*popt),color="red",linewidth=4)
    show()
      
    # print result_____________________________________________
    if(nome=="co"):
        chi=sum( (( _Y-gaus2(_X,*popt) )/_dy)**2 )
    else:
        chi=sum( (( _Y-distr(_X,*popt) )/_dy)**2 )
    
    dof=len(_Y)-len(popt)
#    n,mu,sig,m,q=popt
#    dn,dmu,dsig,dm,dq=sqrt(pcov.diagonal())
#    print("norm=",n,"+-",dn)
#    print("centro=",mu,"+-",dmu," digit")
#    print("largh=",sig,"+-",dsig," digit")
#    print("pendenza=",m,"+-",dm," 1/digit")
#    print("intercetta=",q,"+-",dq)
#    print("")
    print("_____________________________________________\n")
    print("FIT "+el+" riga "+str(energy)+"MeV\n")
    print("chi/ndof=",chi,"+-",sqrt(2*dof)," / ",dof)
    print("p=",chdtrc(dof,chi)*100,"% \n")
    print("result")
    print(lab.format_par_cov(popt,pcov),"\n")