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
import mc

alluminio=[
'0327-all8coll1.txt',
'0412-all8coll5.txt',]

oro0_2=[
'0322-oro0.2coll1.txt',
'0322-oro0.2coll5.txt']

oro5=[
'0320-oro5coll1.txt',
'0419-oro5coll5.txt'
]

varname = 'alluminio'   # immettere il nome del materiale
files = eval(varname)  

print("\n______________ %s ___________\n"%varname.upper())

fig = plt.figure()
fig.clf()
fig.set_tight_layout(True)
plt.title("%s"%varname)
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
        a='black'; nome='1$\!$ mm'
    else:
        a='gray'; nome='5$\!$ mm'
        
    lab4.errorbar(angoli,rate,xerr=1,yerr=unp.sqrt(conteggi)/tempi,linestyle='',color=a,marker='',capsize=0,label='dati collimatore '+nome)

# selezioni

# alluminio [0,0]
# oro3 [-5,6]
# oro5 [-10,15]
tagli = dict(oro5=[-10,15], oro0_2= [-5,6],alluminio= [0,0])
fuori=tagli[varname] # estremi inclusi
vecchio_atot=atot.copy()
vecchio_rtot=rtot.copy()

# selezione per il coll1
y1=[]
for l in range(len(atot[0])):
    if fuori[0]<atot[0][l]<fuori[1]:
        y1.append(l)
atot[0]=np.delete(atot[0],y1)
rtot[0]=np.delete(rtot[0],y1)

mrk=10
if len(y1)>0:
    ax1.plot(nom(vecchio_atot[0][y1]),nom(vecchio_rtot[0][y1]),'xk',label='punti esclusi dal fit',markersize=mrk)

# selezione per il coll5
if len(atot)>1:
    w=np.concatenate((atot[1:]))
    rr=np.concatenate((rtot[1:]))
    
    y5=[]
    for k in range(len(w)):
        if fuori[0]<w[k]<fuori[1] or abs(nom(w[k]))>60:
            y5.append(k)
    w=np.delete(w,y5)
    rr=np.delete(rr,y5)
if len(y5)>0:
    if len(y1)>0:
        ax1.plot(nom(vecchio_atot[1][y5]),nom(vecchio_rtot[1][y5]),'xk',markersize=mrk)
    else:
        ax1.plot(nom(vecchio_atot[1][y5]),nom(vecchio_rtot[1][y5]),'xk',label='punti esclusi dal fit',markersize=mrk)

# fit

def fitfun(teta,A,tc):
    "Rutherford con pdf per il collimatore da 5 mm"
    def f(a, tetax):
        l=28.5 # mm
        d=31
        #np.pi/2+np.arctan(a/d)-np.arcsin( (l*np.cos(tetax))/(np.sqrt(a**2+l**2+2*a*l*np.sin(tetax))) )
        return np.arctan(np.tan(tetax)-a/(l*np.cos(tetax))) - np.arctan(a/d)
    
    def integrando(a,A,tc,tetax):
        return A/( np.sin( (f(a,tetax-tc))/2 ))**4
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

ax1.set_yscale('log')

# monte di carlo
#theta,pesi,energia=mc.mc_cached(seed=0, N=10000000, theta_eps=1,**mc.target_au5,**mc.coll_1)
#pesi/=4e7
#ax1.hist(np.degrees(theta), bins=int(np.sqrt(len(theta))), weights=pesi, histtype='step', density=False)

'''
rappo=fit2.par[0]/fit1.par[0] * (np.sin( (np.radians(nom(w))-fit1.par[1])/2 ))**4 / (np.sin( (np.radians(nom(w))-fit2.par[1])/2 ))**4
rr/=rappo
lab4.errorbar(w-np.degrees(fit2.par[1]),rr,fmt='.g',capsize=2,label='collimatore da 5$\!$ mm scalato')
'''
z1=np.linspace(min(nom(w))-10,max(nom(w)),1000)
ax1.plot(z1,semplice(np.radians(z1),*fit1.par),color='black',linewidth=0.5,scaley=False,label='fit collimatore 1$\!$ mm')


if len(atot)>1:    
    z2=np.linspace(min(nom(w)),-10,1000)
    z3=np.linspace(10,max(nom(w)),1000)
    
    ax1.plot(z2,fitfun(np.radians(z2),*fit2.par),color='gray',linewidth=0.5,linestyle='--',scaley=False,label='fit collimatore 5$\!$ mm')
    ax1.plot(z3,fitfun(np.radians(z3),*fit2.par),color='gray',linewidth=0.5,linestyle='--',scaley=False)



plt.legend(fontsize='small')
fig.show()

print("_______________%s_____________\n"%varname.upper())

print("Rapporto collimatori=",fit2.upar[0]/fit1.upar[0])