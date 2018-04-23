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



l_au=5
l_all=8

n_au=19.32/197  # 1/cm**3
n_all=2.699/27

z_au=79

# scopo del gioco: commentare quello che non serve

# parametri alluminio
# coll1
B_al=uf(5.7,0.1)*1e-6
# coll5
#B_al=uf(1.92,0.14)*1e-5

# parametri oro 3 um
# coll1
#B_au=uf(1.2,0.1)*1e-4
# coll5
#B_au=uf(5.26,0.22)*1e-4

# parametri oro 5 um
# coll1
B_au=uf(1.2,0.1)*1e-4
# coll5
#B_al=uf(6.8,0.3)*1e-4

z_al=z_au*unp.sqrt( B_al/B_au * n_au/n_all * l_au/l_all )

print("Z_alluminio={:1uP}".format(z_al))