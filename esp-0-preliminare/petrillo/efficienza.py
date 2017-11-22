from pylab import *
from scipy.special import expi
import lab

gm = 0.57721566490153286 # euler-mascheroni
    
v=array([i for i in range(1700,2100,100)])

figure('efficienza').set_tight_layout(True)
clf()
grid()

for j in range(len(v)):
    S = loadtxt("eff_%dV.txt" % v[j], unpack=True)
    
    x = S[:,0] # soglie
    
    y = S[:,2]/S[:,1] # rapporti (efficienze)
    mu = S[:,1] # rate coincidenze a due
    dy = sqrt(y * (1 - y) / (exp(mu) - 1) * (expi(mu) - log(mu) - gm))
    
    errorbar(x, y, yerr=dy, fmt='.--', label="%i V" % v[j])
    
    print("alimentazione pmt3=%i V"%v[j])
    for k in range(len(x)):
        print("{:.1f} mV".format(x[k]), "\t efficienza = %s" % lab.xe(y[k], dy[k]))
    print("\n")
    
title("Efficienza PMT3")
xlabel("soglia (mV)")
ylabel("efficienza")

legend(loc="lower right", fontsize='small')
show()
