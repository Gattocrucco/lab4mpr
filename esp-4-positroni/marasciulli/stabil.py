## EVOLUZIONE TEMPORALE DEGLI ISTOGRAMMI
from pylab import *
import lab4
import lab


file="0503_stab"
cartella="../../DAQ/"
ch1,ch2,ch3,tr1,tr2,tr3,c2,c3,ts=lab4.loadtxt(cartella+file+".txt",unpack=True,usecols=(0,1,2,4,5,6,8,9,12))

pezzi=10
chname='ch1' # sono ammessi: ch1,ch2,ch3 

# dati di calibrazione
out1=ch1[tr1>500]
out2=ch2[tr2>500]
out3=ch3[tr3>500]

ts1=ts[tr1>500]
ts2=ts[tr2>500]
ts3=ts[tr3>500]

figure('tempo')
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
outfile=open("slice_%d_%s.txt"%(pezzi,chname),"w")
outfile.write("FILE: %s \n\n"%file)

for j in range(pezzi):
    slice=canale[int((j/pezzi)*len(canale)) :int(((j+1)/pezzi)*len(canale))]
    
    if j+1<pezzi:
        tempo=(cant[int(((j+1)/pezzi)*len(cant))]-ts[0])/3600
    else:
        tempo=(max(cant)-ts[0])/3600
        
    counts,edges=histogram(slice,bins=arange(0,1200//8)*8)
    line, = lab4.bar(edges-2, counts, label='$\Delta t$=%.1f ore'%tempo)
    color=line.get_color()
    legend()
    
    
    # fit
    def gauss(x, peak, mean, sigma):
        return peak * np.exp(-(x - mean) ** 2 / sigma ** 2)
    
    
    x = (edges[1:] + edges[:-1]) / 2
    
    # picchi
    
    for i in range(2):
        if i==0:
            y=x[x<500]
            count=counts[x<500]
        else:
            y=x[x>500]
            count=counts[x>500]
    
        p0 = [1] * 3
        argmax = np.argmax(count)
        # initial parameters
        p0[0] = count[argmax] # peak
        p0[1] = y[argmax] # mean
        p0[2] = 50 # sigma
        cut = (count > count[argmax] / 3) & (y>350)
        if np.sum(cut) > 1:
            out = lab.fit_curve(gauss, y[cut], count[cut], p0=p0, dy=np.sqrt(count)[cut], print_info=1)
            outputs.append(out)
        else:
            outputs.append(None)
        
        # plot
        if not outputs[-1] is None:
            xspace = np.linspace(np.min(y[cut]), np.max(y[cut]), 1000)
            plot(xspace, gauss(xspace, *out.par), '--',color=color)
        
        if i==0:
            outfile.write("annichilazione {} = {:1u} \t\t".format(j+1,out.upar[1]))
        else:
            outfile.write("neon {} = {:1u} \n".format(j+1,out.upar[1]))
    
    
outfile.close()    
minorticks_on()
show()