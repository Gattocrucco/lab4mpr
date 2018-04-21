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

par_all=(uf(3.7e-05 , 0.3e-5) , uf(0.043 , 0.007))  # coll5
par_oro0_2=(uf(0.00012538143518207068 , 1.078666576167907e-05), uf(0.05670765051091632 , 0.00665115129475818))  # coll1

def rute(teta,A,tc):
    return A/np.sin((teta-tc)/2)**4
 
par_all=tuple(nom(par_all))
sigma=quad(rute,np.radians(3),np.pi,args=par_all)
print("sezione d'urto=",sigma[0])