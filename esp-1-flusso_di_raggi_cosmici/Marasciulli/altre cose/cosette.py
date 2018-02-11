## ANGOLO DI PERDITA

try:
    os.chdir("altre cose")
except FileNotFoundError:
    pass
# from pylab import *
# from uncertainties import ufloat as uf
# from importante import astd
# from scipy.optimize import curve_fit

# dati
piccolo_angolo=["","_II","_III"]   # hanno tutti 'ang_6and1' davanti
grande_angolo=["","_II"]           # hanno davanti 'ang_6and5andnot4' e dietro .dat
cartella="C:/Users/andre/Desktop/ANDREA/Laboratorio 4/flusso cosmici/de0_data/"

i1=array([])
i2=array([])

for i in range(len(piccolo_angolo)):
    si=loadtxt(cartella+"ang_6and1"+piccolo_angolo[i]+".dat",unpack=True,usecols=1)
    i1=append(i1,si)
    
for i in range(len(grande_angolo)):
    si=loadtxt(cartella+"ang_6and5andnot4"+grande_angolo[i]+".dat",unpack=True,usecols=1)
    i2=append(i2,si)


# istogrammi
figure(1).set_tight_layout(True)
rc("font",size=16)
grid(color="black",linestyle=":")
minorticks_on()

title("Perdita di energia ed angolo",size=18)
xlabel("energia ADC  (mV)")
ylabel("occorrenze normalizzate")

hist(i1*1000,normed=True,color="black",rwidth=1,label='"piccolo angolo"',bins="auto")
hist(i2*1000,normed=True,color=[0.8,0.8,0.8],rwidth=0.9,label='"grande angolo"',bins="auto")

legend(fontsize="small",loc="best")
show()

# calcolo delle mode

occ1,edge1=histogram(i1,bins="auto")
occ1=list(occ1)
ind1=occ1.index(max(occ1))
moda1=(edge1[ind1+1]+edge1[ind1])/2
err1=(edge1[ind1+1]-edge1[ind1])/2

occ2,edge2=histogram(i2,bins="auto")
occ2=list(occ2)
ind2=occ2.index(max(occ2))
moda2=(edge2[ind2+1]+edge2[ind2])/2
err2=(edge2[ind2+1]-edge2[ind2])/2

print("moda piccolo angolo=",moda1*1000,"+-",err1*1000," mV")
print("moda grande angolo=",moda2*1000,"+-",err2*1000," mV")

## DOPPIETTO?

digi=i1*2**12/3.3
seq=arange(0,2893,14)+0.5
conv=3.3/2**12

figure(1).set_tight_layout(True)
rc("font",size=16)
grid(color="black",linestyle=":")
minorticks_on()

title("Ipotesi particelle doppie",size=18)
xlabel("energia ADC  (mV)")
ylabel("occorrenze")

occ,bor,no=hist(i1*1000,bins=seq*conv*1000,rwidth=0.9)

show()

mo=moda(digi,binni=seq,per=2)
mod=mo*3.3/2**12
print("moda doppie=",mod," V")


## EFFICIENZA LOCALE

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

figure(2).set_tight_layout(True)
rc("font",size=16)
ax1=subplot(111,projection="3d")

x=[1,2,3,4]*4
y=[1,1,1,1  ,2,2,2,2, 3,3,3,3  ,4,4,4,4]
z=zeros(16)

dx=ones(len(x))*0.3
dy=ones(len(y))*0.3
dz=array([552,627,635,716,  523,546,550,666  ,421,534,485,667,  517,494,537,576])
dezu=dz/max(dz)

ax1.bar3d(x,y,z,dx,dy,dezu,color="cyan")
xlabel("colonna")
ylabel("riga")
ax1.set_zlabel("conteggi normalizzati")

colonne=["D","","C","","B","","A"]
righe=[1,"",2,"",3,"",4]
ax1.w_xaxis.set_ticklabels(colonne)
ax1.w_yaxis.set_ticklabels(righe)
title("Efficienza locale normalizzata",size=18)

show()

conti=uarray(dz,sqrt(dz))
mass=max(med(conti))
norm=conti/mass

## SEGNALE SU RUMORE

cartella="C:/Users/andre/Desktop/ANDREA/Laboratorio 4/flusso cosmici/de0_data/"
v=[r for r in range(1600,2100,100)]


for i in range(len(v)):
    no,adc,non=loadtxt(cartella+"signal_"+str(v[i])+".dat",unpack=True)
    no,cda,non=loadtxt(cartella+"noise_"+str(v[i])+".dat",unpack=True)
    
    figure(v[i]).set_tight_layout(True)
    rc("font",size=16)
    grid(color="black",linestyle=":")
    minorticks_on()

    title("Segnale vs rumore a %d V"%v[i],size=18)
    xlabel("tensione ADC  (V)")
    ylabel("occorrenze normalizzate")
    
    # sturges per 1600 V
    hist(cda,bins="sqrt",color="red",label="rumore",rwidth=0.9,normed=True)
    hist(adc,bins="sqrt",color="blue",label="segnale",rwidth=0.6,normed=True)
    
    legend(fontsize="small")
    show()

# bump 2000 V

no,adc,lol=loadtxt(cartella+"signal_2000.dat",unpack=True)
no,cda,lol=loadtxt(cartella+"noise_2000.dat",unpack=True)
del no,lol

segnale,segn=histogram(adc,bins="sqrt",normed=True)
vero,ver=histogram(adc,bins="sqrt",normed=False)
rumore,rum=histogram(cda,bins="sqrt",normed=True)

hist(cda,bins="sqrt",normed=True,rwidth=0.8)
hist(adc,bins="sqrt",normed=True,rwidth=0.9)

show()
# ho visto dall'istogramma che il bump si trova nei primi 7 canali

n=7
sumj=uf(sum(vero),sqrt(sum(vero)))
sumi=uf(sum(vero[:n]),sqrt(sum(vero[:n])))

R=sumi/sumj

print("importanza bump=",R*100,"%")

