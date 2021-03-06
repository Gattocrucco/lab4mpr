## CALIBRAZIONE CON LE SORGENTI
import pylab as plt
import numpy as np
from pylab import *
from scipy.optimize import curve_fit
from scipy.special import chdtrc
import lab
import copy

cartella ="cal/"
date =["22feb","26feb","27feb"]
elements0= [["cs","Cs 137", [0.662], [3700,4000], 5e2],
           ["am","Am 241", [0.060], [320,380], 1e4],
           ["na","Na 22",  [0.511], [2900,3100],5e2],
           ["na","Na 22",  [1.275], [7100,7400],1e1],
           ["co","Co 60", [1.173,1.333], [6450,7850],1e2]]



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

for data in date:
    #if data != '26feb': continue
    for b in arange(len(elements0)):
         #if b != 3: continue
        elements = copy.deepcopy(elements0)
        a = elements[b]
        ref0=loadtxt(cartella+"histo-22feb-"+a[0]+".dat",unpack=True)
        ref1=loadtxt(cartella+"histo-"+data+"-"+a[0]+".dat",unpack=True)
        ref0_max_ind = np.argmax(ref0)
        ref1_max_ind = np.argmax(ref1)
        ref0_max = max(ref0)
        ref1_max = max(ref1)
        conv0 = ref1_max_ind/ref0_max_ind
        conv1 = ref1_max/ref0_max
        elements[b][3] = asarray(around(array(a[3])*conv0),dtype='u8')
        elements[b][4] = elements[b][4]*conv1
        
        nome=a[0]
        el=a[1]
        file=cartella+"histo-"+data+"-"+nome+".dat"
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
            val=[a[4],(3*sin+dex)/4,(dex-sin)/4,10**2,(sin+3*dex)/4,(dex-sin)/4,0,0]
            popt,pcov=curve_fit(gaus2,_X,_Y,sigma=_dy,p0=val, maxfev=10000)
            val = popt
            sin = abs(int(popt[1]-1.5*abs(popt[2])))
            dex = abs(int(popt[4]+1.8*abs(popt[5])))
            _Y = Y[sin:dex]
            _X = arange(dex-sin)+0.5+sin
            _dy=sqrt(_Y)
            popt,pcov=curve_fit(gaus2,_X,_Y,sigma=_dy,p0=val, maxfev=10000)
            
        else:
            val=[a[4],(sin+dex)/2,(dex-sin)/2,0,0]
            popt,pcov=curve_fit(distr,_X,_Y,sigma=_dy,p0=val, maxfev=10000)
            val = popt
            sin = abs(int(abs(popt[1])-1.8*abs(popt[2])))
            dex = abs(int(abs(popt[1])+1.8*abs(popt[2])))
            _Y = Y[sin:dex]
            _X = arange(dex-sin)+0.5+sin 
            _dy=sqrt(_Y)
            popt,pcov=curve_fit(distr,_X,_Y,sigma=_dy,p0=val, maxfev=10000)
            out = lab.fit_curve(distr,_X,_Y,dy=_dy,p0=val, method="odrpack", pfix=[0,0])
            popt,pcov = out.par, out.cov
            
        
        #plot______________________________________________________
        
        #plot su tutto lo spettro
        figure(data+" "+el+"_spectrum").set_tight_layout(True)
        
        rc("font",size=14)
        title("Calibrazione %s del %s" %(data, el), size=16)
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
        
        figure(data+" "+el+"_"+str(energy[0])+"MeV").set_tight_layout(True)
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
        print("FIT "+data+" "+el+" riga "+str(energy)+"MeV\n")
        print("chi/ndof=",int(chi),"+-",int(sqrt(2*dof))," / ",dof)
        print("p=",chdtrc(dof,chi)*100,"% \n")
        print("result")
        print(lab.format_par_cov(popt,pcov),"\n")
        
        file_print=open("/cal/cal.txt","a")
        if(nome=="co"):
            print("%s \t %s \t %f \t %f \t %f \t %f \t %f \t %f" %(data, el, energy[0],popt[1],popt[2],sqrt(pcov[1][1]),sqrt(pcov[2][2]),pcov[1][2]),file=file_print)
            print("%s \t %s \t %f \t %f \t %f \t %f \t %f \t %f" %(data, el, energy[1],popt[4],popt[5],sqrt(pcov[4][4]),sqrt(pcov[5][5]),pcov[4][5]),file=file_print)
        else:
            print("%s \t %s \t %f \t %f \t %f \t %f \t %f \t %f" %(data, el, energy[0],popt[1],popt[2],sqrt(pcov[1][1]),sqrt(pcov[2][2]),pcov[1][2]),file=file_print)
        file_print.close()
    