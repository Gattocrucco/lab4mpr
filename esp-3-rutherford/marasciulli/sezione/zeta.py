## RAPPORTI Z
import numpy as np
import matplotlib.pyplot as plt
from uncertainties import unumpy as unp
import lab4
import lab
from uncertainties.unumpy import nominal_values as nom
from uncertainties.unumpy import std_devs as err
from scipy.special import chdtrc
from uncertainties import ufloat as uf

# preparazione dati
# per far tornare tutto (almeno adesso) ho commentato i dati a doppio

ang_oro,rate_oro,drate_oro=np.loadtxt('oro0_2.txt',unpack=True)  # cambiare da qui lo spessore
ang_all,rate_all,drate_all=np.loadtxt('alluminio.txt',unpack=True)

ang_oro=unp.uarray(ang_oro,1)
rate_oro=unp.uarray(rate_oro,drate_oro)
ang_all=unp.uarray(ang_all,1)
rate_all=unp.uarray(rate_all,drate_all)

l_au=5
l_all=8

n_au=19.32/197  # 1/cm**3
n_all=2.699/27

z_au=79

# calcoli

ang_au=set(nom(ang_oro))
ang_al=set(nom(ang_all))
scelti=set.intersection(ang_au,ang_al)
scelti=list(scelti)

z_all=np.array([])
for i in range(len(scelti)):
    if (scelti[i] in nom(ang_oro)) and (scelti[i] in nom(ang_all)):
        elemento=z_au*unp.sqrt( n_au/n_all * l_au/l_all * rate_all[nom(ang_all)==scelti[i]]/rate_oro[nom(ang_oro)==scelti[i]] )
        z_all=np.append(z_all,elemento)

# grafico

plt.figure().set_tight_layout(True)
plt.rc('font',size=16)
plt.grid(linestyle=":")
plt.minorticks_on()

plt.title("Z dell'alluminio",size=18)
plt.xlabel("angolo  [°]")
plt.ylabel("Z")

plt.errorbar(scelti,nom(z_all),xerr=1,yerr=err(z_all),fmt='.k')

plt.show()