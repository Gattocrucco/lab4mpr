## MOSTRA BIN
from pylab import *
import lab4

t,val=loadtxt('de0_data/0319stab.dat',unpack=True,usecols=(0,1))

figure().set_tight_layout(True)
rc('font',size=16)
grid(linestyle=':')

hist(val+0.5,bins=arange(4096),histtype='step',color='gray',label='dati grezzi')
ylabel('occorrenze  [digit$^{-1}]$')
xlabel('canali ADC  [digit]')

k=arange(128)
p,bins=histogram(val+0.5,bins=k*32)
lab4.bar(bins,p/32,color='black',linewidth=2,label='dati ribinnati')

legend(fontsize='x-small',loc=0)
minorticks_on()
show()