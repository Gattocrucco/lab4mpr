from pylab import *
from uncertainties import ufloat
from uncertainties.unumpy import uarray
import lab

s3, s4, a3, a4, clock, c3, c4, c34, c62, c652, c6523, c6524 = loadtxt('presadati1212.txt', unpack=True)

tm = ufloat(700e-9, 2e-9) # secondi, tempo morto

time = uarray(clock / 1000, 0.5 / 1000) # secondi, tempo di misura

c3 = uarray(c3, sqrt(c3))
c4 = uarray(c4, sqrt(c4))

c34 = uarray(c34, sqrt(c34))

# tassi totali
r3 = c3 / time
r4 = c4 / time

# efficienze totali
E3 = c6523 / c652
E3 = uarray(E3, sqrt(E3*(1-E3)/c652))
E4 = c6524 / c652
E4 = uarray(E4, sqrt(E4*(1-E4)/c652))

e3 = E3 / (1 - tm * r3)
e4 = E4 / (1 - tm * r4)

# rumore atteso
time_3 = ufloat(38, 2) * 1e-9
time_4 = ufloat(37, 2) * 1e-9

r34_att = (time_3 + time_4) * r3 * r4

# misura
mis = (c34 / time - r34_att) / (E3 * E4) * 7.115

# rumore su 3&4
r_att = (c34 / time - r34_att) / (E3 * E4)

n34 = c34 - r_att[-1] * time * E3 * E4
r34 = n34 / time

print("soglia3\talim3\teff3\t\tsoglia4\talim4\teff4\t\tnoise34\tnoisec\tmis")
for i in range(len(s3)):
    print("%.1f\t%.0f\t%10s\t%.1f\t%.0f\t%10s\t%10s\t%10s\t%10s" % (s3[i], a3[i], lab.xe(e3[i].n, e3[i].s), s4[i], a4[i], lab.xe(e4[i].n, e4[i].s), lab.xe(r34[i].n, r34[i].s), lab.xe(r34_att[i].n, r34_att[i].s), lab.xe(mis[i].n, mis[i].s)))
