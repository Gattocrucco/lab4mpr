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



n_au=19.32/197  # 1/cm**3
n_all=2.699/27

z_au=79

# risultati fit
#
############# al8 coll1
#  5.7(1.1)e-6   41.5 %
#              0.026(8)
# centro vero coll1= 1.5+/-0.5 °
# chi quadro= 3.0846274131655456 +- 2.8284271247461903   dof= 4
# P valore= 0.543764444641643
#
############# al8 coll5
#  2.19(19)e-5    8.8 %
#              0.021(6)
# centro vero coll5= 1.2+/-0.4 °
# chi quadro= 10.698151291506667 +- 4.0   dof= 8
# P valore= 0.21939579562169206
#
############# oro3 coll1
#  1.22(10)e-4    9.1 %
#              0.053(6)
# centro vero coll1= 3.02+/-0.35 °
# chi quadro= 3.8316051125887687 +- 3.7416573867739413   dof= 7
# P valore= 0.7989561508409471
#
############# oro3 coll5
#  5.26(22)e-4    5.8 %
#              0.019(6)
# centro vero coll5= 1.1+/-0.4 °
# chi quadro= 11.070007284746564 +- 4.242640687119285   dof= 9
# P valore= 0.27094081318077473
#
############# oro5 coll1
#  1.21(11)e-4   -1.4 %
#              0.038(7)
# centro vero coll1= 2.2+/-0.4 °
# chi quadro= 7.873735965229816 +- 2.8284271247461903   dof= 4
# P valore= 0.09631445533695639
#
############# oro5 coll5
#  6.8(3)e-4    -6.2 %
#            -0.008(7)
# centro vero coll5= -0.5+/-0.4 °
# chi quadro= 80.530152605063 +- 3.7416573867739413   dof= 7
# P valore= 1.0738947710500088e-14

# scopo del gioco: commentare quello che non serve

# parametri alluminio
# coll1
# B_al=uf(5.7,1.1)*1e-6
# coll5
B_al=uf(2.19, 0.19)*1e-5
l_all=8

# parametri oro 3 um
# coll1
# B_au=uf(1.22, 0.10)*1e-4
# coll5
B_au=uf(5.26, 0.22)*1e-4
l_au=3

# parametri oro 5 um
# coll1
# B_au=uf(1.21, 0.11)*1e-4
# coll5
# B_au=uf(6.8, 0.3)*1e-4
# l_au=5

z_al=z_au*unp.sqrt( B_al/B_au * n_au/n_all * l_au/l_all )

print("Z_alluminio={:1uP}".format(z_al))

# risultati Z
# oro3 coll1 10.4±1.1
# oro3 coll5 9.8±0.5
# oro5 coll1 13.4±1.4
# oro5 coll5 11.1±0.5