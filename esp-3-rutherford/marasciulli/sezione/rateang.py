import numpy as np
import matplotlib.pyplot as plt
from uncertainties import unumpy as unp
import lab4
import lab
from uncertainties.unumpy import nominal_values as nom
from uncertainties.unumpy import std_devs as err
from scipy.special import chdtrc
from uncertainties import ufloat as uf


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
ax1 = fig.add_subplot(111)

atot=[] # set di angoli
rtot=[] # set di rate

for file in files:
    ang, count, clock = np.loadtxt('../dati/{}'.format(file), unpack=True)

    count = unp.uarray(count, np.sqrt(count))
    time = unp.uarray(clock, 0.5) * 1e-3
    rate = count / time
    ang = unp.uarray(ang, 1)

    atot.append(ang)
    rtot.append(rate)
    
    if 'coll1' in file:
        a='blue'
    else:
        a='red'
    lab4.errorbar(ang,rate,capsize=2,label="%s"%file,linestyle='',color=a)


# selezioni

fuori=[-11,15] # estremi inclusi

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
    
    #w=np.delete(w,-1) # correzione fatta a mano ATTENZIONE
    #rr=np.delete(rr,-1)

# fit
# mi serve una selezione con gli angoli

def fitfun(teta,A,tc):
    return A/np.sin((teta-tc)/2)**4

# fit1 e fit5 prendono il nome dai collimatori
val1=[1e-3,np.radians(3)]
fit1=lab.fit_curve(fitfun,np.radians(nom(atot[0])),nom(rtot[0]),dx=err(unp.radians(atot[0])),dy=err(rtot[0]),p0=val1,print_info=1)

print("")
print("centro vero coll1=",unp.degrees(fit1.upar[1]),"°")
dof=len(atot[0])-len(fit1.par)
print("chi quadro=",fit1.chisq,"+-",np.sqrt(2*dof),"  dof=",dof)
print("P valore=",chdtrc(dof,fit1.chisq),"\n")

if len(atot)>1:
    val2=[1e-2,np.radians(10)]
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

z1=np.linspace(min(nom(atot[0])),max(nom(atot[0])),1000)
ax1.plot(z1,fitfun(np.radians(z1),*fit1.par),'b',scaley=False)

if len(atot)>1:
    z2=np.linspace(min(nom(w)),max(nom(w)),1000)
    ax1.set_xlim(nom(min(w))-5,nom(max(w))+5)
    ax1.plot(z2,fitfun(np.radians(z2),*fit2.par),'r',scaley=False)

plt.legend(fontsize='small')
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