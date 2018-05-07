## EVOLUZIONE TEMPORALE DEGLI ISTOGRAMMI
from pylab import *
from lab4 import loadtxt as load


file="0504_3gamma"
cartella="../DAQ/"
ch1,ch2,ch3,tr1,tr2,tr3,c2,c3,ts=load(cartella+file+".txt",unpack=True,usecols=(0,1,2,4,5,6,8,9,12))

pezzi=10
chname='ch3' # sono ammessi: ch1,ch2,ch3 

# dati di calibrazione
out1=ch1[tr1>500]
out2=ch2[tr2>500]
out3=ch3[tr3>500]

ts1=ts[tr1>500]
ts2=ts[tr2>500]
ts3=ts[tr3>500]

figure('tempo')
clf()

title("Scalibrazione")
xlabel("valore ADC [digit]")
ylabel("conteggi")


if chname=='ch1':
    canale=out1; cant=ts1
elif chname=='ch2':
    canale=out2; cant=ts2
else:
    canale=out3; cant=ts3

for j in range(pezzi):
    slice=canale[int((j/pezzi)*len(canale)) :int(((j+1)/pezzi)*len(canale))]
    
    if j+1<pezzi:
        tempo=(cant[int(((j+1)/pezzi)*len(cant))]-ts[0])/3600
    else:
        tempo=(max(cant)-ts[0])/3600
        
    hist(slice,bins=arange(0,1200//8)*8,label='$\Delta t$=%.1f ore'%tempo,histtype='step')
    
    legend()
    
minorticks_on()
show()