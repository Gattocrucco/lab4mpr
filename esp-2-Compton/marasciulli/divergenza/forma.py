from pylab import *
import glob, scanf
import uncertainties as un
from uncertainties import unumpy as unp
import lab

sys.stdout=open("ris_div_2.txt","w")

print("DIVRGENZA ANGOLARE")
print("Divergenza angolare con tutto il collimatore uscito \n")

cut = (7400, 7850)
fitcut = (-5, 5)
angle_un = 0.1

files = glob.glob('../dati/histo-22feb-*gradi-*.dat')
file_fondo = glob.glob('../dati/histo-20feb-daunaltraparte-*.dat')[0]

# legge il fondo
data_fondo = loadtxt(file_fondo, unpack=True)
count_fondo = sum(data_fondo[logical_and(arange(len(data_fondo)) >= cut[0], arange(len(data_fondo)) <= cut[1])])
ucount_fondo = un.ufloat(count_fondo, sqrt(count_fondo), tag='countfondo')
clock_fondo, = scanf.scanf('%*sparte-%d.dat', s=file_fondo)
rate_fondo = ucount_fondo / (clock_fondo * 1e-3)

rates = []
angles = []
for file in files:
    data = loadtxt(file, unpack=True, dtype='int64')
    count = sum(data[logical_and(arange(len(data)) >= cut[0], arange(len(data)) <= cut[1])])
    string = file.replace('m', '-')
    angle, clock = scanf.scanf('-%dgradi-%u.dat', s=string)
    angle = float(angle)
    rate = un.ufloat(count, sqrt(count), tag='countang%.0f' % angle) / (clock * 1e-3) - rate_fondo
    rates.append(rate)
    angles.append(angle)
angles = array(angles)
rates = array(rates)
    
# fit
bool_cut = logical_and(fitcut[0] <= angles, angles < fitcut[1])
fit_angles = angles[bool_cut]
fit_rates = rates[bool_cut]
def fit_fun(angle, top, center, sigma):
    return top * exp(-1/2 * (angle - center)**2 / sigma**2)
out = lab.fit_curve(fit_fun, fit_angles, unp.nominal_values(fit_rates), dx=angle_un, dy=unp.std_devs(fit_rates), absolute_sigma=True, p0=(max(unp.nominal_values(fit_rates)), mean(fit_angles), (max(fit_angles) - min(fit_angles)) / 2))
par = out.par
cov = out.cov

print('risultato del fit (top, center, sigma):')
print(lab.format_par_cov(par, cov))
print('center = %s °' % lab.xe(par[1], sqrt(cov[1,1]), pm=lab.unicode_pm))
print('sigma = %s °' % lab.xe(par[2], sqrt(cov[2,2]), pm=lab.unicode_pm))
print('chi2/dof = %.1f/%d' % (out.chisq, out.chisq_dof))

figure('forma')
clf()
errorbar(angles, unp.nominal_values(rates), xerr=angle_un, yerr=unp.std_devs(rates), fmt=',k', capsize=2, label='dati')
fa = linspace(min(fit_angles), max(fit_angles), 500)
plot(fa, fit_fun(fa, *par), '-', color='blue', zorder=-10, label='fit gaussiano')
ylabel('rate nei canali %d-%d [$s^{-1}$]\n(fondo sottratto = %s $s^{-1}$)' % (cut[0], cut[1], format(rate_fondo)))
xlabel('angolo [°]')
title('forma del fascio')
#yscale('log')
legend(loc=1,fontsize="small")
grid()
tight_layout()
show()

sys.stdout.close()
sys.stdout=sys.__stdout__