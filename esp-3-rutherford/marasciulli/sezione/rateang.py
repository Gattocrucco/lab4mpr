import numpy as np
import matplotlib.pyplot as plt
from uncertainties import unumpy as unp
import lab4
import lab
from uncertainties.unumpy import nominal_values as nom
from uncertainties.unumpy import std_devs as err
from scipy.special import chdtrc
from uncertainties import ufloat as uf
from scipy.integrate import quad

alluminio=[
'0327-all8coll1.txt',
'0412-all8coll5.txt',
'0413-all8coll5.txt']

oro0_2=[
'0322-oro0.2coll1.txt',
'0322-oro0.2coll5.txt']

oro5=[
'0320-oro5coll1.txt',
'0419-oro5coll5.txt'
]

varname = 'oro5'   # immettere il nome del materiale
files = eval(varname)  

print("\n______________ %s ___________\n"%varname.upper())

fig = plt.figure('rateang')
fig.clf()
fig.set_tight_layout(True)
plt.rc('font',size=16)
plt.title("%s"%varname,size=18)
ax1 = fig.add_subplot(111)

atot=[] # set di angoli
rtot=[] # set di rate

for file in files:
    ang, count, clock = np.loadtxt('../dati/{}'.format(file), unpack=True)

    count = unp.uarray(count, np.sqrt(count))
    time = unp.uarray(clock, 0.5) * 1e-3
    ang = unp.uarray(ang, 1)

    # somma doppi
    
    angoli = np.unique(nom(ang))
    conteggi = np.empty(angoli.shape,dtype=object)
    tempi = np.zeros(angoli.shape,dtype=object)

    for i in range(len(angoli)):
        conti = count[nom(ang) == angoli[i]]
        temps=time[nom(ang)==angoli[i]]
        conteggi[i] = sum(conti)
        tempi[i]=sum(temps)
        
    rate=conteggi/tempi
        
    atot.append(unp.uarray(angoli,1))
    rtot.append(rate)

    if 'coll1' in file:
        a='blue'
    else:
        a='red'

# selezioni

fuori=[-6,50] # estremi inclusi

# selezione per il coll1
y1=[]
for l in range(len(atot[0])):
    if fuori[0]<atot[0][l]<fuori[1]:
        y1.append(l)
atot[0]=np.delete(atot[0],y1)
rtot[0]=np.delete(rtot[0],y1)

# selezione per il coll5
if len(atot)>1:
    w=np.concatenate((atot[1:]))
    rr=np.concatenate((rtot[1:]))
    
    y5=[]
    for k in range(len(w)):
        if fuori[0]<w[k]<fuori[1] or nom(w[k])==150.0:
            y5.append(k)
    w=np.delete(w,y5)
    rr=np.delete(rr,y5)

# fit

def fitfun(teta,A,tc):
    "Rutherford con pdf per il collimatore da 5 mm"
    def f(a, tetax):
        l=28.5 # mm
        d=31
        #np.pi/2+np.arctan(a/d)-np.arcsin( (l*np.cos(tetax))/(np.sqrt(a**2+l**2+2*a*l*np.sin(tetax))) )
        return np.arctan(np.tan(tetax)-a/(l*np.cos(tetax))) - np.arctan(a/d)
    
    def integrando(a,A,tc,tetax):
        return A/( np.sin( (f(a,tetax)-tc)/2 ))**4
    amax=2.5
    amin=-2.5
    
    integrali=np.empty(len(teta))
    for x in range(len(teta)):
        integrali[x]=quad(integrando,amin,amax,args=(A,tc,teta[x]))[0]/(amax-amin)
    return integrali

def semplice(teta,A,tc):
    return A/( np.sin( (teta-tc)/2 ) )**4

# fit1 e fit5 prendono il nome dai collimatori
val1=[1e-3,np.radians(3)]
fit1=lab.fit_curve(semplice,np.radians(nom(atot[0])),nom(rtot[0]),dx=err(unp.radians(atot[0])),dy=err(rtot[0]),p0=val1,print_info=1)

print("")
print("centro vero coll1=",unp.degrees(fit1.upar[1]),"°")
dof=len(atot[0])-len(fit1.par)
print("chi quadro=",fit1.chisq,"+-",np.sqrt(2*dof),"  dof=",dof)
print("P valore=",chdtrc(dof,fit1.chisq),"\n")

if len(atot)>1:
    val2=[1e-2,np.radians(1)]
    fit2=lab.fit_curve(fitfun,np.radians(nom(w)),nom(rr),dx=err(unp.radians(w)),dy=err(rr),p0=val2,print_info=1)

    print("")    
    print("centro vero coll5=",unp.degrees(fit2.upar[1]),"°")
    dof=len(w)-len(fit2.par)
    print("chi quadro=",fit2.chisq,"+-",np.sqrt(2*dof),"  dof=",dof)
    print("P valore=",chdtrc(dof,fit2.chisq),"\n")

# grafico    

ax1.set_xlabel('angolo [°]')
ax1.set_ylabel('rate [s$^{-1}$]')
ax1.grid(linestyle=':')


lab4.errorbar(atot[0],rtot[0],fmt='.b',capsize=2,label='collimatore da 1$\!$ mm')
lab4.errorbar(w,rr,fmt='.r',capsize=2,label='collimatore da 5$\!$ mm')
'''
rappo=fit2.par[0]/fit1.par[0] * (np.sin( (np.radians(nom(w))-fit1.par[1])/2 ))**4 / (np.sin( (np.radians(nom(w))-fit2.par[1])/2 ))**4
rr/=rappo
lab4.errorbar(w-np.degrees(fit2.par[1]),rr,fmt='.g',capsize=2,label='collimatore da 5$\!$ mm scalato')
'''
z1=np.linspace(min(nom(w))-10,max(nom(w)),1000)
ax1.plot(z1,semplice(np.radians(z1),*fit1.par),'b',scaley=False)


if len(atot)>1:    
    z2=np.linspace(min(nom(w)),-10,1000)
    z3=np.linspace(15,max(nom(w)),1000)
    
    ax1.plot(z2,fitfun(np.radians(z2),*fit2.par),'r',scaley=False)
    ax1.plot(z3,fitfun(np.radians(z3),*fit2.par),'r',scaley=False)



plt.legend(fontsize='x-small')
fig.show()

print("_______________%s_____________\n"%varname.upper())



'''
# scrviere i rate sul file
f=open('%s.txt'%varname,'w')
print("# dati %s"%varname,file=f)
print("# angolo[°]\trate[Hz]\terrore rate",file=f)

for i in range(len(atot)):
    for j in range(len(atot[i])):
        print("%d\t\t%f\t%f"%(nom(atot[i][j]),nom(rtot[i][j]),err(rtot[i][j])),file=f)
        
f.close()
'''