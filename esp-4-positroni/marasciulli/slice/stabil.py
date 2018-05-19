## EVOLUZIONE TEMPORALE DEGLI ISTOGRAMMI
from pylab import *
import lab4
import lab
from fit_peak import fit_peak
import gvar


file="0503_stab"
cartella="../DAQ/"
ch1,ch2,ch3,tr1,tr2,tr3,c2,c3,ts=lab4.loadtxt(cartella+file+".txt",unpack=True,usecols=(0,1,2,4,5,6,8,9,12))

pezzi=4
chname='ch2' # sono ammessi: ch1,ch2,ch3 

# dati di calibrazione
out1=ch1[(ch1>150) & (ch2<150)]
out2=ch2[(ch1<150) & (ch2>150)]
out3=ch3[ch3>150]

ts1=ts[(ch1>150) & (ch2<150)]
ts2=ts[(ch1<150) & (ch2>150)]
ts3=ts[ch3>150]

figure('tempo').set_tight_layout(True)
clf()

title("Scalibrazione del %s in %s"%(chname,file))
xlabel("valore ADC [digit]")
ylabel("conteggi")


if chname=='ch1':
    canale=out1; cant=ts1
elif chname=='ch2':
    canale=out2; cant=ts2
else:
    canale=out3; cant=ts3
    
outputs=[]
beta=array([])
neon=array([])
tempi=array([])

for j in range(pezzi):
    slice=canale[int((j/pezzi)*len(canale)) :int(((j+1)/pezzi)*len(canale))]
    
    if j+1<pezzi:
        tempo=(cant[int(((j+1)/pezzi)*len(cant))]-ts[0])/3600
    else:
        tempo=(max(cant)-ts[0])/3600
        
    tempi=append(tempi,tempo)
        
    counts,edges=histogram(slice,bins=arange(0,1200//8)*8)
    line, = lab4.bar(edges, counts, label='$\Delta t$=%.1f ore'%tempo)
    color=line.get_color()
    legend()
    
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
            outdict,indict = fit_peak(bordi,ordinata,bkg='exp',npeaks=1,ax=gca(),cut=cut,plot_kw={'color':color})
        else:
            outdict,indict = fit_peak(bordi,ordinata,bkg=None,npeaks=1,ax=gca(),cut=cut,plot_kw={'color':color})
    

'''
figure('grafico').set_tight_layout(True)
clf()

title("Scalibrazione del %s in %s"%(chname,file))
ylabel("valore ADC [digit]")
xlabel("tempo  [ore]")
grid(linestyle=':')
minorticks_on()

subplot(211)
lab4.errorbar(tempi,beta,fmt='.r',label='Annichilazione')
legend(loc=0)
grid(linestyle=':')
minorticks_on()

subplot(212)
lab4.errorbar(tempi,neon,fmt='.g',label='Neon')

grid(linestyle=':')
minorticks_on()

legend(loc='lower right')
minorticks_on()
show()
'''