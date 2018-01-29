## ANGOLO DI PERDITA

try:
    os.chdir("altre cose")
except FileNotFoundError:
    pass
# from pylab import *
# dati
piccolo_angolo=["","_II","_III"]   # hanno tutti 'ang_6and1' davanti
grande_angolo=["","_II"]           # hanno davanti 'ang_6and5andnot4' e dietro .dat
cartella="C:/Users/andre/Desktop/ANDREA/Laboratorio 4/flusso cosmici/de0_data/"

i1=array([])
i2=array([])

for i in range(len(piccolo_angolo)):
    no,si,non=loadtxt(cartella+"ang_6and1"+piccolo_angolo[i]+".dat",unpack=True)
    i1=append(i1,si)
    
for i in range(len(grande_angolo)):
    no,si,non=loadtxt(cartella+"ang_6and5andnot4"+grande_angolo[i]+".dat",unpack=True)
    i2=append(i2,si)
    
del no,non

# istogrammi
figure(1).set_tight_layout(True)
rc("font",size=16)
title("Perdita di energia ed angolo",size=18)
grid(color="black",linestyle=":")
minorticks_on()

xlabel("energia ADC  (mV)")
ylabel("occorrenze normalizzate")

hist(i1*1000,normed=True,color="blue",rwidth=1,label="piccolo angolo",bins="auto")
hist(i2*1000,normed=True,color="orange",rwidth=0.9,label="grande angolo",bins="auto")

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