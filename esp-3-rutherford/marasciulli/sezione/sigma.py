## INTEGRAZIONE SEZIONE D'URTO DIFFERENZIALE
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

par_all=(uf(3.529676144312048e-05 , 2.5886646789266666e-06) , uf(0.030456451504531253 , 0.008192846854198087))  # coll5
par_oro0_2=(uf(0.00012538143518207068 , 1.078666576167907e-05), uf(0.05670765051091632 , 0.00665115129475818))  # coll1

def rute(teta,A,tc):
    return A/np.sin((teta-tc)/2)**4
 
par_oro0_2=tuple(nom(par_oro0_2))
sigma=quad(rute,np.radians(15),np.pi,args=par_oro0_2)
print("sezione d'urto=",sigma[0])