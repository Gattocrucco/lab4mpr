## VERIFICA DELLA STATISTICA POISSONIANA
try:
    os.chdir("esperienza preliminare")
except FileNotFoundError:
    pass

print("ESPERIENZA 0 \n")

from math import factorial
# dati
n=py.loadtxt("pm2 1s.txt",unpack=True)

x=array([])
occ=array([])
for numero in n:
    if numero not in x:
        x=py.append(x,numero)
        k=0
        for j in range(len(n)):
            if n[j]==numero:
                k+=1
        occ=py.append(occ,k)
  
def fattoriale(vettore):
    "Fattoriale che funziona solo sui vettori"
    fac=array([])
    for el in vettore:
        le=factorial(int(el))
        fac=py.append(fac,le)
    return fac

# fit
def poissoniana(x,mu):
    return mu**x*e**(-mu)/fattoriale(x)
"""    
stime=py.mean(n)
popt,pcov=curve_fit(poissoniana,x,occ,stime) #non vuole funzionare
mu_f=popt
dmu_f=sqrt(pcov.diagonal())
print("media=",mu_f,"+-",dmu_f)
"""

# grafico
py.figure(1)
py.rc("font",size=16)
py.grid(color="black",linestyle="--")
py.minorticks_on()

py.title("Conteggio PM2 caldo",size=18)
py.xlabel("eventi osservati")
py.ylabel("occorrenze")

#py.bar(x,poissoniana(x,25)*py.sum(occ),color="orange",edgecolor="black",width=0.6)

py.bar(x,occ,color="",edgecolor="blue")

py.tight_layout()
py.show()