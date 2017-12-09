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

asc=array([30,20,10,0])  # correggere
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

py.errorbar(asc,med(ord),xerr=0.1,yerr=err(ord),linestyle="",color="red",capsize=2,marker=".")
z=py.linspace(-0.5,30.5,10**3)
py.plot(z,att(z,*popt),color="blue")

py.show()

chi=py.sum( ((med(ord)-att(asc,*popt))/err(ord))**2 )
dof=len(ord)-len(popt)
p=chdtrc(dof,chi)
print("chi quadro=",chi,"+-",sqrt(dof))
print("p_value=",p)

#sys.stdout.close()
#sys.stdout=sys.__stdout__

## LUNGHEZZA DI ATTENUAZIONE PER RIGHE

#sys.stdout=open("mr.txt","w")

print("MEDIA PER RIGHE \n")
# Faccio la media dei conteggi di ogni riga e poi li fitto

c=py.loadtxt("numeri_griglia.txt")
x=array([30,20,10,0])  # correggere
col=["red","green","blue","orange"]

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
    
    stime=[600,100]
    popt,pcov=curve_fit(att,x,y,stime,sigma=dy)
    Ai,li=popt
    dAi,dli=sqrt(pcov.diagonal())
    
    A=py.append(A,Ai)
    dA=py.append(dA,dAi)
    
    l=py.append(l,li)
    dl=py.append(dl,dli)
    
    py.errorbar(x,y,xerr=0.5,yerr=dy,marker="+",linestyle="",capsize=2,label="riga %d" %(4-i), color=col[i] )
    z=py.linspace(-0.5,30.5,1000)
    py.plot(z,att(z,*popt),color=col[i])
    
    chi=py.sum( ((y-att(x,*popt))/dy)**2 )
    dof=len(y)-len(popt)
    p=chdtrc(dof,chi)
    print("chi quadro riga %d=" %(4-i),chi,"+-",sqrt(dof))
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