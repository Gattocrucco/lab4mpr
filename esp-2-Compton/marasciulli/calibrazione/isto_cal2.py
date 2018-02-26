## CALIBRAZIONE CON LE SORGENTI
import lab

cartella="dati/"
data="22feb-"
nome="co-trigger"; noto=1.332
file=cartella+"histo-"+data+nome+".dat"
el="cobalto"
sce="con" # scegliere tra 'con' e 'senza'

scrivi=False
if scrivi==True:
    sys.stdout=open("calibrazione/%s_%s_trigger.txt"%(el,sce),"w")

grezzi=loadtxt(file,unpack=True)

# creazione istogramma
print("CALIBRAZIONE \n")

nbin=350
massimo=8192
lbin=massimo//nbin+1

dati=zeros(nbin)
for j in range(len(grezzi)):
    indice=j//lbin
    dati[int(indice)]+=grezzi[j]

conv=massimo/len(dati)
X=arange(len(dati))*conv

sin=6240
dex=7621
taglio=logical_and(X>=sin,X<=dex)

# fit

def distr(x,N,u,sigma,m,q):
    return N*e**(-1*((x-u)**2)/(2*sigma**2)) + m*x+q
    
def gaus2(x,N1,u1,sigma1,N2,u2,sigma2,m,q):
    return N1*e**(-1*((x-u1)**2)/(2*sigma1**2)) + N2*e**(-1*((x-u2)**2)/(2*sigma2**2)) +m*x+q

dy=sqrt(dati[taglio])

val=[10**3,6930,300,10**3,7833,300,0,10**3]
popt,pcov=curve_fit(gaus2,X[taglio],dati[taglio],sigma=dy,p0=val)
n1,mu1,sig1,n2,mu2,sig2,m,q=popt
dn1,dmu1,dsig1,dn2,dmu2,dsig2,dm,dq=sqrt(pcov.diagonal())


fis=noto/mu2

# grafico

figure(1).set_tight_layout(True)
rc("font",size=14)
title("Calibrazione %s %s trigger"%(el,sce),size=16)
grid(color="black",linestyle=":")
minorticks_on()

xlabel("energia [MeV]")
ylabel("conteggi")

bar(X*fis,dati,width=lbin*fis)
z=linspace(sin,dex,1000)
plot(z*fis,gaus2(z,*popt),color="red",linewidth=2)
show()

# altro
chi=sum( (( dati[taglio]-gaus2(X[taglio],*popt) )/dy)**2 )
dof=len(dati[taglio])-len(popt)

print("chi quadro=",chi,"+-",sqrt(2*dof))
print("p=",chdtrc(dof,chi)*100,"% \n")
print("matrice di covarianza")
print(lab.format_par_cov(popt,pcov),"\n")

print("picco 1=",mu1*fis,"+-",dmu1*fis,"MeV")
print("largh 1=",sig1*fis,"+-",dsig1*fis,"MeV")
print("picco 2=",mu2*fis,"+-",dmu2*fis,"MeV")
print("largh 2=",sig2*fis,"+-",dsig2*fis,"MeV")

# cose da scrivere nel file della linearità
'''
registro=open("calibrazione/autoestratti.txt","a")
print("%f \t %f \t %f \t %f \t %f \t %f" %(noto,mu,dmu,sig,dsig,cor(1,2,pcov)),file=registro )
registro.close()
'''

if scrivi==True:
    sys.stdout.close()
    sys.stdout=sys.__stdout__