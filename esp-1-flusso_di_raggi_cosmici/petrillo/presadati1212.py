from pylab import *
from uncertainties import ufloat
from uncertainties.unumpy import uarray
import lab
from montecarlo import MC, pmt

s3, s4, a3, a4, clock, c3, c4, c34, c62, c652, c6523, c6524 = loadtxt('presadati1212.txt', unpack=True)

time = uarray(clock / 1000, 0.5 / 1000) # secondi, tempo di misura

# errori sui conteggi
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
E5 = c652 / c62
E5 = uarray(E5, sqrt(E5*(1-E5)/c62))

# efficienze intrinseche
tm = ufloat(700e-9, 2e-9) # secondi, tempo morto

e3 = E3 / (1 - tm * r3 / (1 - tm * r3))
e4 = E4 / (1 - tm * r4 / (1 - tm * r4))
#ACHTUNG! uncertainties correla totalmente tm in e3 e e4, ma sono due tempi morti diversi, misurati indipendentemente.

# rumore atteso
time_3 = ufloat(38, 2) * 1e-9
time_4 = ufloat(37, 2) * 1e-9

n34_att = (time_3 + time_4) * r3 * r4

# misura
mis = (c34 / time - n34_att) / (E3 * E4) * 7.115

# rumore su 3&4
# calcolato in modo che su una delle misure sia uguale a quello atteso
r_att = (c34 / time - n34_att) / (E3 * E4)

n34 = c34 / time - r_att[-1] * E3 * E4

print("%8s%6s%13s%9s%6s%13s%12s%12s%12s" % ("soglia3","alim3","eff3","soglia4","alim4","eff4","noise34","noisec","mis"))
for i in range(len(s3)):
    print("%8.1f%6.0f%13s%9.1f%6.0f%13s%12s%12s%12s" % (s3[i], a3[i], lab.xe(e3[i].n, e3[i].s), s4[i], a4[i], lab.xe(e4[i].n, e4[i].s), lab.xe(n34[i].n, n34[i].s), lab.xe(n34_att[i].n, n34_att[i].s), lab.xe(mis[i].n, mis[i].s)))
    
def corr(distcos=None):
    mc = MC(pmt(5),pmt(4),pmt(3))
    mc.random_ray(N=1e5, distcos=distcos)
    mc.run(pivot_scint=2)
    
    rlon = mc.density(1,None,1) * (mis / 7.115)[-1]
    rvic = mc.density(None,1,1) * (mis / 7.115)[-2]
    
    print('vicini  ', rvic)
    print('lontani ', rlon)
