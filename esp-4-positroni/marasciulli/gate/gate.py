## EFFETTO DELLA DURATA DEL GATE
from pylab import *
import sys
import lab4
import lab
from fit_peak_originale import fit_peak
import gvar
import glob
import scanf

files = glob.glob('../DAQ/0504_gate*_*.txt')

durata= array([])
mark=array([])
beta= array([])
sbeta = array([])
insieme=set()

gra=figure('')
gra.set_tight_layout(True)
rc('font',size=14)

ax=gra.add_subplot(111)

for i in range(len(files)):
    files[i] = files[i].replace("\\", '/')
    ch1,ch2,ch3,tr1,tr2,tr3,c2,c3,ts = lab4.loadtxt(files[i], unpack=True, usecols=(0,1,2,4,5,6,8,9,12))
    durata= append(durata , scanf.scanf('../DAQ/0504_gate%d_%d.txt', s=files[i])[0])
    rit=scanf.scanf('../DAQ/0504_gate%d_%d.txt', s=files[i])[1]
    
    if rit==0:
        mark=append(mark,'.')
    elif rit==16:
        mark=append(mark,'v')
    elif rit==32:
        mark=append(mark,'^')
    elif rit==40:
        mark=append(mark,'x')
    
    out1=ch1[(ch1>250) & (ch2<150)]
    
    isto=figure('%s'%files[i])
    mi=isto.add_subplot(111)
    counts,edges,_=hist(out1,bins=arange(0,1200//8)*8,histtype='step')
    
    X=edges[1:]-edges[:1]/2
    
    bordi=edges[edges<800]
    count=counts[X<800]

    argmax = np.argmax(count) 
    if '400' in files[i]:
        sin=60; dex=40;
    else:
        sin=30; dex=40;
    cut = (bordi[argmax]-sin,bordi[argmax]+dex)
    ordinata=gvar.gvar(count,sqrt(count))

    print("_______%s________"%files[i])
    if '300_16' in files[i]:
        fondo='exp'
    else:
        fondo='exp'
    outdict,indict = fit_peak(bordi,ordinata,bkg=fondo,npeaks=1,cut=cut,print_info=1,ax=mi)
    beta=append(beta,outdict['peak1_mean'])
    sbeta=append(sbeta,outdict['peak1_sigma'])
    
    mi.legend()
    

    ax.grid(linestyle=':')
    ax.minorticks_on()
    
    ax.set_title('Variazione del gate')
    ax.set_xlabel('durata del gate  [ns]')
    ax.set_ylabel('$\sigma$/media')
    
    if rit in insieme:
        ax.errorbar(durata[i],gvar.mean(sbeta/beta)[i],gvar.sdev(sbeta/beta)[i],color='blue',capsize=2,linestyle='',marker=mark[i])
    else:
        ax.errorbar(durata[i],gvar.mean(sbeta/beta)[i],gvar.sdev(sbeta/beta)[i],color='blue',capsize=4,linestyle='',marker=mark[i],label='ritardo= %d ns'%rit)
        insieme.add(rit)

ax.legend(loc=0,fontsize='x-small')
show()

# tabella per la relazione

lista=[]
tdurata=sort(durata)
tbeta=beta[argsort(durata)]
tsbeta=tbeta[argsort(durata)]

for j in range(len(tbeta)):
    lista.append([int(tdurata[j]),'{}'.format(tbeta[j]),'{}'.format(tsbeta[j]),'{}'.format(tsbeta[j]/tbeta[j])])
    
print(lab.TextMatrix(lista).latex())