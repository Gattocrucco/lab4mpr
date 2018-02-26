## CALIBRAZIONE CON LE SORGENTI
import lab

cartella="dati/"
data="22feb-"
nome="ce-notrigger"; noto=0.662
file=cartella+"histo-"+data+nome+".dat"
el="cobalto"
sce="senza" # scegliere tra 'con' e 'senza'

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

sin=3470
dex=4000
taglio=logical_and(X>=sin,X<=dex)

# fit

def distr(x,N,u,sigma,m,q):
    return N*e**(-1*((x-u)**2)/(2*sigma**2)) + m*x+q
    
def gaus2(x,N1,u1,sigma1,N2,u2,sigma2):
    return N1*e**(-1*((x-u1)**2)/(2*sigma1**2)) + N2*e**(-1*((x-u2)**2)/(2*sigma2**2))

dy=sqrt(dati[taglio])

val=[10**4,(sin+dex)/2,(dex-sin)/2,1,1]
popt,pcov=curve_fit(distr,X[taglio]+0.5,dati[taglio],sigma=dy,p0=val)
n,mu,sig,m,q=popt
dn,dmu,dsig,dm,dq=sqrt(pcov.diagonal())
print("norm=",n,"+-",dn)
print("centro=",mu,"+-",dmu," digit")
print("largh=",sig,"+-",dsig," digit")
print("pendenza=",m,"+-",dm," 1/digit")
print("intercetta=",q,"+-",dq)
print("")

fis=noto/mu

# grafico

figure(1).set_tight_layout(True)
rc("font",size=14)
title("Calibrazione %s %s trigger"%(el,sce),size=16)
grid(color="black",linestyle=":")
minorticks_on()

xlabel("energia [MeV]")
ylabel("conteggi")

bar(X*fis,dati,width=lbin*fis)
#errorbar(X[taglio],dati[taglio],yerr=dy,color="black",linestyle="",capsize=2,markersize=2,marker="o")
z=linspace(sin,dex,1000)
plot(z*fis,distr(z,*popt),color="red",linewidth=2)
show()

# altro
chi=sum( (( dati[taglio]-distr(X[taglio],*popt) )/dy)**2 )
dof=len(dati[taglio])-len(popt)

print("chi quadro=",chi,"+-",sqrt(2*dof))
print("p=",chdtrc(dof,chi)*100,"% \n")
print("matrice di covarianza")
print(lab.format_par_cov(popt,pcov),"\n")

print("centro=",mu*fis,"+-",dmu*fis,"MeV")
print("largh=",sig*fis,"+-",dsig*fis,"MeV")
print("pendenza=",m/fis,"+-",dm/fis,"1/MeV")

# cose da scrivere nel file della linearità
'''
registro=open("calibrazione/autoestratti.txt","a")
print("%f \t %f \t %f \t %f \t %f \t %f" %(noto,mu,dmu,sig,dsig,cor(1,2,pcov)),file=registro )
registro.close()
'''
if scrivi==True:
    sys.stdout.close()
    sys.stdout=sys.__stdout__