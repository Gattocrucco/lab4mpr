import numpy as np
import matplotlib.pyplot as plt
import uncertainties as un
from uncertainties import unumpy as unp
import lab4
import os
from scipy.special import chdtrc
import lab

###### RATE vs. ANGLE ######

data_prefixs   = ['0316-forma']
spectr_prefixs = ['0316ang']
colls          = [False]

fig = plt.figure('forma')
fig.clf()
fig.set_tight_layout(True)
plt.rc("font",size=14)
plt.rc("errorbar",capsize=2)

ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)

for data_prefix, spectr_prefix, coll in zip(data_prefixs, spectr_prefixs, colls):
    ang, count, time = np.loadtxt('../dati/{}.txt'.format(data_prefix), unpack=True)
    ang_labels = np.loadtxt('../dati/{}.txt'.format(data_prefix), usecols=(0,), dtype=str)
    diameter = un.ufloat(16.3, 0.1)
    overhang = un.ufloat(5.3, 0.1)
    source = un.ufloat(3.1, 0.1)
    L = diameter / 2 - overhang # cm
    D = source # cm

    time = unp.uarray(time, 0.5) * 1e-3
    count = unp.uarray(count, np.sqrt(count))

    if not coll:
        # vedi sul logbook <<angolo forma>> per questi calcoli
        theta = unp.radians(unp.uarray(ang, 1))
        X = unp.sqrt(L**2 + 2*L*D*unp.cos(theta) + D**2)
        alpha = np.sign(unp.nominal_values(theta)) * unp.arccos((L*unp.cos(theta) + D) / X)

        # theta = 0 va propagato a parte
        theta_0 = theta[unp.nominal_values(theta) == 0]
        alpha_0 = L/(L + D) * theta_0
        alpha[unp.nominal_values(theta) == 0] = alpha_0
        alpha = unp.degrees(alpha)
    
        rate = count / time * X**2
    else:
        alpha = unp.uarray(ang, 0)
        rate = count / time


    ###### SPECTRUM vs. ANGLE ######

    s, cl, lengths = np.empty((3, len(ang)))
    serr = np.empty((2, len(ang)))
    for i in range(len(ang)):
        angle = ang[i]
        filename = '../de0_data/{}{}.dat'.format(spectr_prefix, ang_labels[i].replace('-', '_'))
        if angle == 80 and spectr_prefix == '0316ang':
            filename = '../de0_data/0319ang80new.dat'
        if not os.path.exists(filename):
            print('warning: file {} does not exist'.format(filename))
            s[i] = np.nan
            serr[:, i] = np.nan
            cl[i] = np.nan
            lengths[i] = np.nan
            continue
        print('processing {}...'.format(filename))
        t, ch1, ch2 = np.loadtxt(filename, unpack=True)
        out = lab4.credible_interval(ch1, cl=0.9)
        s[i] = out[0]
        serr[0, i] = out[0] - out[1]
        serr[1, i] = out[2] - out[0]
        cl[i] = out[3]
        lengths[i] = len(ch1)
        if abs(len(ch1) - count[i].n) / count[i].n > 0.001:
            print('warning: angle {:.2f}: ADC and scaler counts differ more than 0.1 %'.format(angle))
            
    ###### PLOT ######

    lab4.errorbar(alpha, rate, ax=ax1, fmt='.', markersize=4, label=data_prefix,capsize=2,color="black")
    lab4.errorbar(alpha, s, yerr=serr, ax=ax2, fmt='.', markersize=4, label=data_prefix, capsize=2,color="black")
    
    # mi sembra una parabola
    def parabola(x,a,b,c):
        return a*x**2+b*x+c
    
    stime=[2,1,1600]
    out=lab.fit_curve(parabola,unp.nominal_values(alpha),unp.nominal_values(rate),dx=unp.std_devs(alpha),dy=unp.std_devs(rate),p0=stime,print_info=1,absolute_sigma=True)
    
    z=np.linspace(-45,48,1000)
    ax1.plot(z,parabola(z,*out.par),color="red")
    print("")
    dof=len(alpha)-len(out.par)
    print("chi quadro=", out.chisq,"+-",np.sqrt(2*dof))
    
    p=chdtrc(dof,out.chisq)
    print("P valore=",p,"  dof=",dof)
    
# subplot1
ax1.set_title("Forma del fascio",size=16)

#ax1.legend(loc='upper left', fontsize='small')
ax1.grid(linestyle=':')
ax1.set_ylabel('rate [s$^{-1}$ cm$^{2}$]' if not coll else 'rate [s$^{-1}$]')
ax1.minorticks_on()

#subplot 2

ax2.grid(linestyle=':')
ax2.set_xlabel('angolo [°]')
ax2.set_ylabel('moda (90 % CR) [digit]')
ax2.minorticks_on()

fig.show()
