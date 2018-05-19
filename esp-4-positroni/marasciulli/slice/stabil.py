## EVOLUZIONE TEMPORALE DEGLI ISTOGRAMMI
from pylab import *
import lab4
import lab
from fit_peak import fit_peak
import gvar
from matplotlib.ticker import FuncFormatter


file="0503_stab"
cartella="../DAQ/"
estensione=".txt"

pezzi=25

if estensione=='.txt':
    ch1,ch2,ch3,tr1,tr2,tr3,c2,c3,ts=lab4.loadtxt(cartella+file+estensione,unpack=True,usecols=(0,1,2,4,5,6,8,9,12))
    
    out1=ch1[(ch1>150) & (ch2<150)]
    out2=ch2[(ch1<150) & (ch2>150)]
    out3=ch3[ch3>150]
    
    ts1=ts[(ch1>150) & (ch2<150)]
    ts2=ts[(ch1<150) & (ch2>150)]
    ts3=ts[ch3>150]
    
    p=3
    righe=2
    colonne=3
    
elif estensione=='.npz':
    ogg=load(cartella+file+estensione)
    ch1=ogg['ch1']
    ts=ogg['ts']
    
    out1=ch1[ch1>150]
    ts1=ts[ch1>150]
    
    p=1
    righe=2
    colonne=1


#figure('tempo').set_tight_layout(True)
#clf()

for z in range(p):
    if z==0:
        canale=out1; cant=ts1
    if z==1:
        canale=out2; cant=ts2
    if z==2:
        canale=out3; cant=ts3
        
        
    
    #subplot(1,3,z+1)

    outputs=[]
    beta=array([])
    sbeta=array([])
    neon=array([])
    sneon=array([])
    tempi=array([])
    
    for j in range(pezzi):
        slice=canale[int((j/pezzi)*len(canale)) :int(((j+1)/pezzi)*len(canale))]
        
        if j+1<pezzi:
            tempo=(cant[int(((j+1)/pezzi)*len(cant))]-ts[0])/3600
        else:
            tempo=(max(cant)-ts[0])/3600
            
        tempi=append(tempi,tempo)
            
        counts,edges=histogram(slice,bins=arange(0,1200//8)*8)
        #line, = lab4.bar(edges, counts, label='$\Delta t$=%.1f ore'%tempo)
        #color=line.get_color()
        #legend()
        
        X=edges[1:]-edges[:1]/2
        
        # fit
        def gauss(x, peak, mean, sigma):
            return peak * np.exp(-(x - mean) ** 2 / sigma ** 2)
        
        for i in range(2):
            if i==0:
                bordi=edges[edges<=500]
                count=counts[X<500]
            else:
                bordi=edges[edges>=496]
                count=counts[X>500]
            
            argmax = np.argmax(count)               
            cut = (bordi[argmax]-40,bordi[argmax]+40)
            ordinata=gvar.gvar(count,sqrt(count))
            
            if i==0:
                outdict,indict = fit_peak(bordi,ordinata,bkg='exp',npeaks=1,cut=cut)
                beta=append(beta,outdict['peak1_mean'])
                sbeta=append(sbeta,outdict['peak1_sigma'])
            else:
                outdict,indict = fit_peak(bordi,ordinata,bkg=None,npeaks=1,cut=cut)
                neon=append(neon,outdict['peak1_mean'])
                sneon=append(sneon,outdict['peak1_sigma'])
    
    tempi+=19    
    
    figure('grafico').set_tight_layout(False)
    rc('font',size=12)
    grid(linestyle=':')
    minorticks_on()
    
    subplot(righe,colonne,z+1,sharex=gca())
    if z==0:
        ylabel("valore beta [digit]")
    
    gca().xaxis.set_major_formatter(FuncFormatter(lambda x,k: '%g' % (x%24)))
    
    errorbar(tempi,gvar.mean(beta),gvar.sdev(beta),fmt='.r',capsize=2,markersize=3,label='ch%d beta'%(z+1))
    grid(linestyle=':')
    minorticks_on()
    legend(loc=0,fontsize='small')
    
    if colonne==1:
        posto=2
    else:
        posto=z+4
        
    subplot(righe,colonne,posto,sharex=gca())
    errorbar(tempi,gvar.mean(neon),gvar.sdev(neon),fmt='.g',capsize=2,markersize=3,label='ch%d neon'%(z+1))
    if z==0:
        ylabel("valore neon [digit]")
    xlabel("orario")
    legend(loc=0,fontsize='small')
    
    grid(linestyle=':')
    minorticks_on()
    gca().xaxis.set_major_formatter(FuncFormatter(lambda x,k: '%g' % (x%24)))
    
    
    figure('rette').set_tight_layout(True)
    
    subplot(righe,colonne,z+1,sharex=gca())
    
    grid(linestyle=':')
    minorticks_on()
    if z==0:
        ylabel("m  [keV/digit]")
    gca().xaxis.set_major_formatter(FuncFormatter(lambda x,k: '%g' % (x%24)))

    q=(1270*beta-511*neon)/(beta-neon)
    m=(511-q)/beta
    errorbar(tempi,gvar.mean(m),gvar.sdev(m),fmt='.k',capsize=2,markersize=3,label='ch%d beta'%(z+1))
    legend(loc=0,fontsize='small')
    
    if colonne==1:
        posto=2
    else:
        posto=z+4
        
    subplot(righe,colonne,posto,sharex=gca())
    xlabel("orario")
    if z==0:
        ylabel("q  [keV]")
    gca().xaxis.set_major_formatter(FuncFormatter(lambda x,k: '%g' % (x%24)))
    
    errorbar(tempi,gvar.mean(q),gvar.sdev(q),fmt='.k',capsize=2,markersize=3,label='ch%d neon'%(z+1))
    legend(loc=0,fontsize='small')
    grid(linestyle=':')
    minorticks_on()
    
    show()
