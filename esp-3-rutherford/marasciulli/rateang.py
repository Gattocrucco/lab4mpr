import numpy as np
import matplotlib.pyplot as plt
from uncertainties import unumpy as unp
import lab4
import lab
from uncertainties.unumpy import nominal_values as nom
from uncertainties.unumpy import std_devs as err

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
    sto supponendo che nel passaggio tra collimatori l'ultimo angolo del primo file
    sia il primo angolo del secondo
    '''
    if 'coll1' in file:
        ultimo=rate[-1]
    if 'coll5' in file:
        rapp=rate[0]/ultimo
        rate/=rapp
    
    angoli=np.append(angoli,ang)
    tassi=np.append(tassi,rate)

ax1.set_xlabel('angolo [°]')
ax1.set_ylabel('rate [s$^{-1}$]')
ax1.grid(linestyle=':')

ax1.errorbar(nom(angoli),nom(tassi),xerr=err(angoli),yerr=err(tassi),capsize=2,marker=".",linestyle="")

fig.show()
