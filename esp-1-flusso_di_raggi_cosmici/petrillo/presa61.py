from pylab import *
from uncertainties import ufloat
from uncertainties.unumpy import uarray, nominal_values
import lab
from montecarlo import MC, pmt

s1, s6, a1, a6, clock, c1, c6, c61, c52, c652, c521, c6521 = loadtxt('presa61.txt', unpack=True)

time = uarray(clock / 1000, 0.5 / 1000) # secondi, tempo di misura

# errori sui conteggi
c1 = uarray(c1, sqrt(c1))
c6 = uarray(c6, sqrt(c6))
c61 = uarray(c61, sqrt(c61))

# tassi totali
r1 = c1 / time
r6 = c6 / time

# efficienze totali
E1 = c6521 / c652
E1 = uarray(E1, sqrt(E1*(1-E1)/c652)) * 2.55
E6 = c6521 / c521
E6 = uarray(E6, sqrt(E6*(1-E6)/c652)) * 1.25

# efficienze intrinseche
tm = ufloat(700e-9, 2e-9) # secondi, tempo morto

e1 = E1 / (1 - tm * r1 / (1 - tm * r1))
e6 = E6 / (1 - tm * r6 / (1 - tm * r6))
#ACHTUNG! uncertainties correla totalmente tm in e3 e e4, ma sono due tempi morti diversi, misurati indipendentemente.

# rumore atteso
time_1 = ufloat(38, 2) * 1e-9
time_6 = ufloat(37, 2) * 1e-9

n61_att = (time_1 + time_6) * r1 * r6

# misura
mis = (c61 / time - n61_att) / (E1 * E6) * 45.96

# rumore su 3&4
# calcolato in modo che su una delle misure sia uguale a quello atteso
r_att = (c61 / time - n61_att) / (E1 * E6)

n61 = c61 / time - r_att[-1] * E1 * E6

print("%8s%6s%13s%9s%6s%13s%12s%12s%12s" % ("soglia1","alim1","eff1","soglia6","alim6","eff6","noise61","noise_c","mis"))
for i in range(len(s1)):
    print("%8.1f%6.0f%13s%9.1f%6.0f%13s%12s%12s%12s" % (s1[i], a1[i], lab.xe(e1[i].n, e1[i].s), s6[i], a6[i], lab.xe(e6[i].n, e6[i].s), lab.xe(n61[i].n, n61[i].s), lab.xe(n61_att[i].n, n61_att[i].s), lab.xe(mis[i].n, mis[i].s)))
    
# def corr(distcos=None):
#     mc = MC(pmt(5),pmt(4),pmt(3))
#     mc.random_ray(N=1e5, distcos=distcos)
#     mc.run(pivot_scint=2)
#
#     rlon = mc.density(1,None,1) * (mis / 7.115)[-1]
#     rvic = mc.density(None,1,1) * (mis / 7.115)[-2]
#
#     print('vicini  ', rvic)
#     print('lontani ', rlon)
