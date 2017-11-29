from pylab import *
from scipy.special import expi
from matplotlib.gridspec import GridSpec
import lab

gm = 0.57721566490153286 # euler-mascheroni
    
v = [1700, 1800, 1900, 2000]

colors = [3*[0.], 3*[0.7], 3*[0.3], 3*[0.5]]
shapes = ['^', 'v', '.', '.']

figure('efficienza').set_tight_layout(True)
clf()
figure('SN').set_tight_layout(True)
clf()

G = GridSpec(3, 1)

sum_c24 = 0

for j in range(len(v)):
    thr, c4, c3, c2, c24, c234 = loadtxt("efficienza_%d.txt" % (v[j],), unpack=True)
    
    eff = c234 / c24
    deff = sqrt(eff * (1 - eff) / (exp(c24) - 1) * (expi(c24) - log(c24) - gm))
    
    sum_c24 += sum(c24)
    if v[j] == 1800:
        eff_24 = eff[1]
        deff_24 = deff[1]
    
    s = c234 / eff**2
    n = c3 - s
    sn = s/n
    
    dc234 = sqrt(c234)
    ds = s * sqrt((dc234/c234)**2 + 2 * (deff/eff)**2)
    dc3 = sqrt(c3)
    dn = sqrt(dc3**2 + ds**2)
    dsn = sn * sqrt((ds/s)**2 + (dn/n)**2)
    
    figure('efficienza')
    errorbar(-thr, eff, yerr=deff, fmt='--.', label="%d V" % (v[j],), color=colors[j])
    
    figure('SN')
    subplot(G[:2,:])
    errorbar(-thr[sn>0], sn[sn>0], yerr=dsn[sn>0], fmt='--%s' % shapes[j], label="%d V" % (v[j],), color=colors[j])
    subplot(G[-1,:])
    errorbar(-thr[sn<=0], sn[sn<=0], yerr=dsn[sn<=0], fmt='--%s' % shapes[j], label="%d V" % (v[j],), color=colors[j])

R = sum_c24 / (eff_24**2 * 160)
dR = R * sqrt(1/sum_c24 + 2*(deff_24/eff_24)**2)
print("R = %s" % lab.xe(R, dR, pm=lab.unicode_pm))
        
figure('efficienza')
grid()
title("Efficienza PMT3")
xlabel("soglia (mV)")
ylabel("efficienza")
legend(loc="lower right", fontsize='small')

figure('SN').set_tight_layout(True)

subplot(G[:2,:])
grid()
title("segnale/rumore PMT3")
# xlabel("soglia (mV)")
ylabel("S/N")
yscale('log')
legend(loc=0, fontsize='small')
lims = xlim()

subplot(G[-1,:])
grid()
# title("segnale/rumore PMT3")
xlabel("soglia (mV)")
ylabel("S/N")
# legend(loc=0, fontsize='small')
xlim(lims)

show()
