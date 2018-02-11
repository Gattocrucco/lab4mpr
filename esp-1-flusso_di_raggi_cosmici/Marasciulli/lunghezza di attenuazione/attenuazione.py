## LUNGHEZZA DI ATTENUAZION PER COLONNE
try:
    os.chdir("lunghezza di attenuazione")
except FileNotFoundError:
    pass

import pylab as py 
#sys.stdout=open("mc.txt","w")

print("MEDIA PER COLONNE \n")
# Faccio la media dei conteggi di ogni colonna e poi li fitto

def att(x,A,l):
    return A*e**(-x/l)
    
D,C,B,A=py.loadtxt("numeri_griglia.txt",unpack=True)

D=uarray( D,sqrt(D) )
C=uarray( C,sqrt(C) )
B=uarray( B,sqrt(B) )
A=uarray( A,sqrt(A) )

asc=array([32.5,23,13.7,3.5])-3.5
ord=array([py.mean(D),py.mean(C),py.mean(B),py.mean(A)])

valori=[666,100]
popt,pcov=curve_fit(att,asc,med(ord),valori,sigma=err(ord))
A,l=popt
dA,dl=sqrt(pcov.diagonal())
print("ampiezza=",A,"+-",dA)
print("lambda=",l,"+-",dl," cm")
print("correlazione=",cov(0,1,pcov),"\n")


py.figure(1).set_tight_layout(True)
py.rc("font",size=16)
py.grid(color="black",linestyle=":")
py.minorticks_on()

py.title("Lunghezza di attenuazione",size=18)
py.xlabel("distanza dalla guida di luce  (cm)")
py.ylabel("conteggi PMT1")

py.errorbar(asc,med(ord),xerr=0,yerr=err(ord),linestyle="",color="red",capsize=2,marker=".")
z=py.linspace(-0.5,33,10**3)
py.plot(z,att(z,*popt),color="blue")

py.show()

chi=py.sum( ((med(ord)-att(asc,*popt))/err(ord))**2 )
dof=len(ord)-len(popt)
p=chdtrc(dof,chi)
print("chi quadro=",chi,"+-",sqrt(2*dof))
print("p_value=",p,"\n")

#sys.stdout.close()
#sys.stdout=sys.__stdout__

## LUNGHEZZA DI ATTENUAZIONE PER RIGHE

#sys.stdout=open("mr.txt","w")

print("MEDIA PER RIGHE \n")
# Faccio la media dei conteggi di ogni riga e poi li fitto

c=py.loadtxt("numeri_griglia.txt")
x=array([32.5,23,13.7,3.5])-3.5 
col=["red","green","blue","orange"]
stile=["--","-","-","-"]
lista=["o","v","^","x"]    # marker

py.figure(2).set_tight_layout(True)
py.rc("font",size=16)
py.grid(color="black",linestyle=":")
py.minorticks_on()

py.title("Lunghezza di attenuazione",size=18)
py.xlabel("distanza dalla guida di luce  (cm)")
py.ylabel("conteggi")

A=array([]); dA=array([])
l=array([]); dl=array([])
matrici=[]; parametri=[]

for i in range(len(c)):
    y=c[i]
    dy=sqrt(y)
    
    # fit
    stime=[600,100]
    popt,pcov=curve_fit(att,x,y,stime,sigma=dy)
    Ai,li=popt
    dAi,dli=sqrt(pcov.diagonal())
    
    matrici.append(pcov)
    parametri.append(popt)
    
    # grafico
    A=py.append(A,Ai)
    dA=py.append(dA,dAi)
    
    l=py.append(l,li)
    dl=py.append(dl,dli)
    
    py.errorbar(x,y,xerr=0,yerr=dy,marker=lista[i],linestyle="",capsize=2,label="riga %d" %(4-i), color=col[i], markersize=8 )
    z=py.linspace(-0.5,33,1000)
    py.plot(z,att(z,*popt),color=col[i],linestyle=stile[i])
    # altri risultati
    
    chi=py.sum( ((y-att(x,*popt))/dy)**2 )
    dof=len(y)-len(popt)
    p=chdtrc(dof,chi)
    print("chi quadro riga %d=" %(4-i),chi,"+-",sqrt(2*dof))
    print("p_value=",p,"\n")

    
    
py.legend(loc="best",fontsize="small")
py.show()

# veri errori e parametri

pezzo1=[]
pezzo2=[]
for i in range(len(matrici)):
    pezzo1.append(inv(matrici[i]))
    pezzo2.append(inv(matrici[i])@parametri[i])
    
pezzo1=py.sum(pezzo1,axis=0)
pezzo2=py.sum(pezzo2,axis=0)

V=inv(pezzo1)
pm=V@pezzo2

# vera correlazione medie

corV=V[0][1]/( sqrt(V[0][0]*V[1][1]) )
errA=sqrt(V[0][0])
errl=sqrt(V[1][1])

print("ampiezza media=",pm[0],"+-",errA)
print("lunghezza media=",pm[1],"+-",errl)
print("corr(A,lambda)=",corV,"\n")

#sys.stdout.close()
#sys.stdout=sys.__stdout__

## ATTENUAZIONE ENERGETICA COLONNE
#sys.stdout=open("ec.txt","w")

print("ATTENUAZIONE ENERGIA COLONNE \n")
import statistics as stat
# dati
colonna=["A","B","C","D"]
riga=[1,2,3,4]
Y=uarray([],[])

for j in range(len(colonna)):
    
    mode=array([])
    er=array([])
    
    for i in range(len(riga)):

        no,en,non=py.loadtxt("C:/Users/andre/Desktop/ANDREA/Laboratorio 4/flusso cosmici/de0_data/misura_%s%s.dat" %(colonna[j],riga[i]),unpack=True)
        del no,non;
        
        t=py.histogram(en,bins="auto")
        tt=list(t[0])
        indice=tt.index(max(tt))
        moda=(t[1][indice]+t[1][indice+1])/2
        mode=py.append(mode,moda)
        errore=(t[1][indice+1]-t[1][indice])/2
        er=py.append(er,errore)

    y=py.average(mode)
    dy=astd(er)  
    Y=py.append(Y,uf(y,dy))

# fit

x=array([3.5,13.7,23,32.5])-3.5
Y*=1000
# grafico 

py.figure(3).set_tight_layout(True)
py.rc("font",size=16)
py.grid(color="black",linestyle=":")
py.minorticks_on()

py.title("Lunghezza di attenuazione",size=18)
py.xlabel("distanza dalla guida di luce  (cm)")
py.ylabel("valore ADC  (mV)")
    
    
py.errorbar(x,med(Y),xerr=0.2,yerr=err(Y),capsize=2,linestyle="",color="black")  # mostro i risultati in mV

    
py.show()


#sys.stdout.close()
#sys.stdout=sys.__stdout__

## ATTENUAZIONE ENERGETICA RIGHE

#sys.stdout=open("er.txt","w")

print("ATTENUAZIONE ENERGIA RIGHE \n")
import statistics as stat

colonna=["A","B","C","D"]
riga=[1,2,3,4]
X=array([3.5,13.7,23,32.5])-3.5
col=["red","green","blue","orange"]
lista=["o","x","^","v"]    # marker

#apertura grafico
py.figure(4).set_tight_layout(True)
py.rc("font",size=16)
py.grid(color="black",linestyle=":")
py.minorticks_on()

py.title("Lunghezza di attenuazione",size=18)
py.xlabel("distanza dalla guida di luce  (cm)")
py.ylabel("valore ADC  (mV)")


for i in range(len(riga)):
    
    mode=array([])
    orr=array([])
    
    for j in range(len(colonna)):
        no,en,non=py.loadtxt("C:/Users/andre/Desktop/ANDREA/Laboratorio 4/flusso cosmici/de0_data/misura_%s%s.dat" %(colonna[j],riga[i]),unpack=True)
        del no,non
        
        h=py.histogram(en,bins="auto")
        hh=list(h[0])
        indice=hh.index(max(hh))
        moda=(h[1][indice]+h[1][indice+1])/2
        mode=py.append(mode,moda)
        orrore=(h[1][indice+1]-h[1][indice])/2
        orr=append(orr,orrore)
    
    mode*=1000
    orr*=1000

    py.errorbar(X+i/1.5,mode,xerr=0,yerr=orr,linestyle="",capsize=2,label="riga %s"%riga[i],marker=lista[i],color=col[i],markersize=7)
    

py.legend(fontsize="x-small",loc="lower left")
py.show()

#sys.stdout.close()
#sys.stdout=sys.__stdout__