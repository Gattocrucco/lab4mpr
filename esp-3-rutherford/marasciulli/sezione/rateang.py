import numpy as np
import matplotlib.pyplot as plt
from uncertainties import unumpy as unp
import lab4
import lab
from uncertainties.unumpy import nominal_values as nom
from uncertainties.unumpy import std_devs as err
from scipy.special import chdtrc

files = [
    '0322-oro0.2coll1.txt',
    '0322-oro0.2coll5.txt',
]

fig = plt.figure('rateang')
fig.clf()
fig.set_tight_layout(True)

ax1 = fig.add_subplot(111)

# array per il concatenate

angoli=np.array([])
tassi=np.array([])

for file in files:
    ang, count, clock = np.loadtxt('../dati/{}'.format(file), unpack=True)

    count = unp.uarray(count, np.sqrt(count))
    time = unp.uarray(clock, 0.5) * 1e-3
    rate = count / time
    ang = unp.uarray(ang, 1)
    '''
    sto supponendo che nel passaggio tra collimatori l'ultimo angolo 
    del primo file sia il primo angolo del secondo
    '''
    if 'coll1' in file:
        ultimo=rate[-1]
    if 'coll5' in file:
        rapp=rate[0]/ultimo
        rate/=rapp
    
    angoli=np.append(angoli,ang)
    tassi=np.append(tassi,rate)
    
# selezione

sel=[-10,15]  # estremi compresi
angf=np.array([]) # f sta per fit
ratef=np.array([]) 

for i in range(len(angoli)):
    if angoli[i]<=sel[0] or angoli[i]>=sel[1]:
        angf=np.append(angf,angoli[i])
        ratef=np.append(ratef,tassi[i])
    else:
        pass

# fit semplice

def rute(teta,A,c):
    return A/(np.sin((teta-c)/2))**4  

p0=[1,np.radians(3.5)]
# ricordatevi gli angoli in radianti
fit=lab.fit_curve(rute,np.radians(nom(angf)),nom(ratef),dx=np.radians(err(angf)),dy=err(ratef),p0=p0,print_info=1)

dof=len(angf)-len(fit.par)
print("chi quadro=",fit.chisq,"+-",sqrt(2*dof),"  dof=",dof)

# grafico    

ax1.set_xlabel('angolo [°]')
ax1.set_ylabel('rate [s$^{-1}$]')
ax1.grid(linestyle=':')

lab4.errorbar(angoli,tassi,capsize=2,marker=".",linestyle="")  # non posso fare ax1.lab4.errorbar

z1=np.linspace(-70,-5,100)
ax1.plot(z1,rute(np.radians(z1),*fit.par),color="red")

z2=np.linspace(10,70,100)
ax1.plot(z2,rute(np.radians(z2),*fit.par),color="red")

fig.show()
