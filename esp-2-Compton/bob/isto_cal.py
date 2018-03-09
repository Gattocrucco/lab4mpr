## CALIBRAZIONE CON LE SORGENTI
import pylab as plt
import numpy as np
from pylab import *
from scipy.optimize import curve_fit
from scipy.special import chdtrc
import lab
import copy

cartella ="cal/"
date =["20feb", "22feb","26feb","27feb"]
elements0= [["cs","Cs137", [0.662], [3700,4000], 5e2],
           ["am","Am241", [0.0595], [320,380], 1e4],
           ["na","Na22",  [0.511], [2900,3100],5e2],
           ["na","Na22",  [1.275], [7100,7400],1e1],
           ["co","Co60", [1.173,1.333], [6450,7850],1e2]]

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
    
def fondo(x,m,q):
    return (m*x+q +abs(m*x+q))/2

#____________________________________________________
file_print=open("cal/cal.txt","w")
print("#data elemento energia [Mev] media[digit] sigma[digit] err_media[digit] err_sigma[digit] cov(media,sigma)",file=file_print)
        
              
for data in date:
    for b in arange(len(elements0)):
        # if b != 4 or data != '20feb': continue
        if data == '20feb' and b != 4:
            continue
        elements = copy.deepcopy(elements0)
        a = elements[b]
        ref0=loadtxt(cartella+"histo-22feb-"+a[0]+".dat",unpack=True)
        if data == '20feb':
            ref1=loadtxt("../dati/histo-20feb-ang0-31471.dat",unpack=True)
        else:
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
        
        Y = ref1
        X = arange(len(Y)+1)
        
        _Y = Y[sin:dex]
        _X = arange(dex-sin)+0.5+sin
        _dy=sqrt(_Y)
        
        # creazione istogramma   
        if(nome == "co"):
            val=[a[4],(3*sin+dex)/4,(dex-sin)/4,10**2,(sin+3*dex)/4,(dex-sin)/4,0,0]
            popt,pcov=curve_fit(gaus2,_X,_Y,sigma=_dy,p0=val, maxfev=10000)
            val = popt
            sin = abs(int(popt[1]-1.6*abs(popt[2])))
            dex = abs(int(popt[4]+1.6*abs(popt[5])))
            _Y = Y[sin:dex]
            _X = arange(dex-sin)+0.5+sin
            _dy=sqrt(_Y)
            out = lab.fit_curve(gaus2,_X,_Y,dy=_dy,p0=val, method="odrpack")
            popt,pcov = out.par, out.cov
            
        else:
            val=[a[4],(sin+dex)/2,(dex-sin)/2,0,0]
            popt,pcov=curve_fit(distr,_X,_Y,sigma=_dy,p0=val, maxfev=10000)
            val = popt
            sin = abs(int(abs(popt[1])-1.6*abs(popt[2])))
            dex = abs(int(abs(popt[1])+1.6*abs(popt[2])))
            _Y = Y[sin:dex]
            _X = arange(dex-sin)+0.5+sin 
            _dy=sqrt(_Y)
            #popt,pcov=curve_fit(distr,_X,_Y,sigma=_dy,p0=val, maxfev=10000)
            out = lab.fit_curve(distr,_X,_Y,dy=_dy,p0=val, method="odrpack")
            popt,pcov = out.par, out.cov
            
        
        #plot______________________________________________________
        
        #plot su tutto lo spettro
        figure(data+" "+el+"_spectrum",figsize=(9, 4), dpi=150).set_tight_layout(True)
        
        rc("font",size=14)
        title("Calibrazione %s del %s" %(data, el), size=16)
        grid(color="black",linestyle=":")
        minorticks_on()
        xlabel("energia [digit]")
        ylabel("conteggi")
        
        #bar(X*fis,Y,width=lbin*fis)
        if(energy!=elements[3][2]):
            k=32
            n_bin = len(Y)//k
            Y_rebin = np.zeros(n_bin)
            for i in arange(n_bin):
                mean=0
                for j in arange(k):
                    mean += Y[i*k+j]
                Y_rebin[i] = mean/k
            X_rebin = arange(n_bin+1)*k
            bar_line(X_rebin, Y_rebin, linewidth=1)
        
        z=linspace(sin,dex,1000)
        if(nome=="co"):
            plot(z,gaus2(z,*popt),color="red",linewidth=1)
            #plot(z,fondo(z,popt[6],popt[7]))
        else:
            plot(z,distr(z,*popt),color="red",linewidth=1)
            #plot(z,fondo(z,popt[3],popt[4]), color="blue")
        savefig("cal/plot/"+data+"_"+el+".pdf")
        savefig("cal/plot/"+data+"_"+el+".png")
        
        #zoom sulla gaussiana    
        
        figure(data+" "+el+"_"+str(energy[0])+"MeV", figsize=(9, 4), dpi=150).set_tight_layout(True)
        rc("font",size=14)
        title("Fit fotopicco a "+str(energy)+"MeV nello spettro del "+el, size=16)
        grid(color="black",linestyle=":")
        minorticks_on()
        xlabel("energia [digit]")
        ylabel("conteggi")
        
        errorbar(_X,_Y, _dy, linestyle="", marker=".", markersize=2, color="black",linewidth=1)
        if(nome=="co"):
            plot(z,gaus2(z,*popt),color="red",linewidth=2)
        else:
            plot(z,distr(z,*popt),color="red",linewidth=2)
        savefig("cal/plot/"+data+"_"+el+"_"+str(energy[0])+".pdf")
        savefig("cal/plot/"+data+"_"+el+"_"+str(energy[0])+".png")
          
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
        
        if(nome=="co"):
            print("%s \t %s \t %f \t %f \t %f \t %f \t %f \t %f \t %f" %(data, el, energy[0],popt[1],abs(popt[2]),sqrt(pcov[1][1]),sqrt(pcov[2][2]),pcov[1][2], pcov[1,4]),file=file_print)
            print("%s \t %s \t %f \t %f \t %f \t %f \t %f \t %f \t %f" %(data, el, energy[1],popt[4],abs(popt[5]),sqrt(pcov[4][4]),sqrt(pcov[5][5]),pcov[4][5], 0),file=file_print)
        else:
            print("%s \t %s \t %f \t %f \t %f \t %f \t %f \t %f \t %f" %(data, el, energy[0],popt[1],abs(popt[2]),sqrt(pcov[1][1]),sqrt(pcov[2][2]),pcov[1][2], 0),file=file_print)

file_print.close()
    