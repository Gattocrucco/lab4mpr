## PARTICELLE DOPPIE
try:
    os.chdir("altre cose")
except FileNotFoundError:
    pass
    
# from pylab import *

cartella="C:/Users/andre/Desktop/ANDREA/Laboratorio 4/flusso cosmici/de0_data/"
soglie=["40_4","100_4","150_8","200_7","248_5","302_8","355_4","413"]
sogliola=array([40.4,100.4,150.8,200.7,248.5,302.8,355.4,413])
mode=array([])

figure(1).set_tight_layout(True)
rc("font",size=16)
grid(color="black",linestyle=":")
minorticks_on()

title("Particelle doppie",size=18)
xlabel("soglia (mV)")
ylabel("moda  (mV)")

for j in range(len(soglie)):
    no,en,non=loadtxt(cartella+"corr_"+soglie[j]+".dat",unpack=True)
    #hist(en,bins="auto",rwidth=0.4,label=soglie[j]+" mV")
    mode=append(mode,moda(en))

mode*=1000
errorbar(sogliola,med(mode),xerr=0.3,yerr=err(mode),linestyle="",capsize=2,color="brown")
    
legend(fontsize="small",loc="best")
show()