## LUNGHEZZA DI ATTENUAZION PER COLONNE
try:
    os.chdir("lunghezza di attenuazione")
except FileNotFoundError:
    pass
    
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

py.errorbar(asc,med(ord),xerr=0.2,yerr=err(ord),linestyle="",color="red",capsize=2,marker="")
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
lista=["o","x","^",""]    # marker

py.figure(2).set_tight_layout(True)
py.rc("font",size=16)
py.grid(color="black",linestyle=":")
py.minorticks_on()

py.title("Lunghezza di attenuazione",size=18)
py.xlabel("distanza dalla guida di luce  (cm)")
py.ylabel("conteggi")

A=array([]); dA=array([])
l=array([]); dl=array([])

for i in range(len(c)):
    y=c[i]
    dy=sqrt(y)
    # fit
    stime=[600,100]
    popt,pcov=curve_fit(att,x,y,stime,sigma=dy)
    Ai,li=popt
    dAi,dli=sqrt(pcov.diagonal())
    # grafico
    A=py.append(A,Ai)
    dA=py.append(dA,dAi)
    
    l=py.append(l,li)
    dl=py.append(dl,dli)
    
    py.errorbar(x,y,xerr=0.2,yerr=dy,marker=lista[i],linestyle="",capsize=2,label="riga %d" %(4-i), color=col[i] )
    z=py.linspace(-0.5,33,1000)
    py.plot(z,att(z,*popt),color=col[i])
    # altri risultati
    chi=py.sum( ((y-att(x,*popt))/dy)**2 )
    dof=len(y)-len(popt)
    p=chdtrc(dof,chi)
    print("chi quadro riga %d=" %(4-i),chi,"+-",sqrt(2*dof))
    print("p_value=",p,"\n")
    
py.legend(loc="best",fontsize="small")
py.show()

Am=py.average(A,weights=1/dA**2)
dAm=astd(dA)
lm=py.average(l,weights=1/dl**2)
dlm=astd(dl)

print("Ampiezza media=",Am,"+-",dAm)
print("lunghezza di attenuazione media=",lm,"+-",dlm," cm")

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
    
    for i in range(len(riga)):

        no,en,non=py.loadtxt("C:/Users/andre/Desktop/ANDREA/Laboratorio 4/flusso cosmici/de0_data/misura_%s%s.dat" %(colonna[j],riga[i]),unpack=True)
        del no,non;
        
        try:
            mode=py.append(stat.mode(en),mode)
        except stat.StatisticsError:
            bt,bb=py.unique(en,return_counts=True)
            BB=list(bb)
            valore=en[BB.index(max(BB))]
            mode=py.append(mode,valore)
        #er=py.append(py.std(en,ddof=1),er)/sqrt(len(en))
    
    y=py.average(mode)
    dy=astd([3/2**12]*4)   # errore di digitalizzazione in volt
    Y=py.append(Y,uf(y,dy))
    
# fit

x=array([3.5,13.7,23,32.5])-3.5
Y*=1000
delta=sqrt( err(Y)**2 + 0.2**2*( A/l*e**(-x/l) )**2 )

def att(x,A,l):
    return A*e**(-x/l)

valori=[300,100]
popt,pcov=curve_fit(att,x[1:],med(Y[1:]),valori,sigma=delta[1:])
A,l=popt
dA,dl=sqrt(pcov.diagonal())
print("ampiezza=",A,"+-",dA," mV")
print("lambda=",l,"+-",dl," cm")
print("correlazione=",cov(0,1,pcov),"\n")

# grafico 

py.figure(3).set_tight_layout(True)
py.rc("font",size=16)
py.grid(color="black",linestyle=":")
py.minorticks_on()

py.title("Lunghezza di attenuazione",size=18)
py.xlabel("distanza dalla guida di luce  (cm)")
py.ylabel("valore ADC  (mV)")
    
    
py.errorbar(x,med(Y),xerr=0.2,yerr=delta,capsize=2,linestyle="",color="black")  # mostro i risultati in mV
z=py.linspace(0,30,10**3)
py.plot(z,att(z,*popt),color="blue")
    
py.show()

chi=py.sum( ((med(Y[1:])-att(x[1:],*popt))/delta[1:])**2 )
dof=len(x[1:])-len(popt)
p=chdtrc(dof,chi)
print("chi quadro=",chi,"+-",sqrt(2*dof))
print("p_value=",p,"\n")

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
lista=["o","x","^",""]    # marker

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
    
    for j in range(len(colonna)):
        no,en,non=py.loadtxt("C:/Users/andre/Desktop/ANDREA/Laboratorio 4/flusso cosmici/de0_data/misura_%s%s.dat" %(colonna[j],riga[i]),unpack=True)
        del no,non
        
        try:
            mode=py.append(stat.mode(en),mode)
        except stat.StatisticsError:
            bt,bb=py.unique(en,return_counts=True)
            BB=list(bb)
            valore=en[BB.index(max(BB))]
            mode=py.append(mode,valore)  
    
    orrore=3/2**12
    '''
    # fit
    # todo: delta
    def att(x,A,l):
        return A*e**(-x/l)
    
    valori=[300,100]
    popt,pcov=curve_fit(att,X,mode,valori,sigma=[orrore]*len(X))
    A,l=popt
    dA,dl=sqrt(pcov.diagonal())
    print("ampiezza=",A,"+-",dA," mV")
    print("lambda=",l,"+-",dl," cm")
    print("correlazione=",cov(0,1,pcov),"\n")
    '''
    # grafico
    
    py.errorbar(X,mode,xerr=0.2,yerr=orrore,linestyle="",capsize=2,label="riga %d"%riga[i],marker=lista[i],color=col[i])
    z=py.linspace(0,30,1000)
    #py.plot(z,att(z,*popt))
    
    chi=py.sum( ((mode-att(X,*popt))/orrore)**2 )
    dof=len(mode)-len(popt)
    p=chdtrc(dof,chi)
    print("chi quadro riga %d=" %(i+1),chi,"+-",sqrt(2*dof))
    print("p_value=",p,"\n")
    
py.legend(fontsize="small",loc="best")
py.show()

#sys.stdout.close()
#sys.stdout=sys.__stdout__