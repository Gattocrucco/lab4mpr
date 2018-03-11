import mc9
import histo
from mc9 import mc_cached


N=100000
p, s, wp, ws = mc_cached(1.33, theta_0=15, N=N, seed=34,date="27feb")
p7, s7, wp7, ws7 = mc_cached(1.17, theta_0=15, N=N, seed=23,date="27feb")
ws/=6; ws7/=6

from matplotlib.pyplot import *
figure('mc9')
clf()

# dati

grezzi=loadtxt("../dati/histo-27feb-e15.dat")
totale=sum(grezzi)
histo.bar_line(arange(8193),grezzi)

# montecarli
wp*=totale/sum(wp)
ws*=totale/sum(ws)
hist(p, bins=int(np.sqrt(N)), weights=wp*sqrt(N)/totale, histtype='step', label='1.33 MeV',color="orange")
hist(s, bins=int(np.sqrt(N)), weights=ws*sqrt(N)/totale, histtype='step', label='spalla 1.33 MeV',color="orange")

wp7*=totale/sum(wp7)
ws7*=totale/sum(ws7)
hist(p7, bins=int(np.sqrt(N)), weights=wp7*sqrt(N)/totale, histtype='step', label='1.17 MeV',color="blue")
hist(s7, bins=int(np.sqrt(N)), weights=ws7*sqrt(N)/totale, histtype='step', label='spalla 1.17 MeV',color="blue")

legend(loc='best')
show()