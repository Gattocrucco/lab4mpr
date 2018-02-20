from pylab import *
import glob, scanf
import uncertainties as un
from uncertainties import unumpy as unp
import optimize

cut = (7400, 7850)
fitcut = (-5, 5)

files = glob.glob('../histo/histo-20feb-ang*-?????.dat')
file_fondo = glob.glob('../histo/histo-20feb-daunaltraparte-*.dat')[0]

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
    string = file.replace('_', '-')
    angle, clock = scanf.scanf('%*s-ang%d-%u.dat', s=string)
    angle = float(angle)
    rate = un.ufloat(count, sqrt(count), tag='countang%.0f' % angle) / (clock * 1e-3) - rate_fondo
    rates.append(rate)
    angles.append(angle)
angles = array(angles)
rates = array(rates)
    
# fit
bool_cut = logical_and(fitcut[0] <= angles, angles <= fitcut[1])
fit_angles = angles[bool_cut]
fit_rates = rates[bool_cut]
def fit_fun(angle, norm, center, sigma):
    return norm * exp(-1/2 * (angle - center)**2 / sigma**2)
par, cov = optimize.curve_fit(fit_fun, fit_angles, unp.nominal_values(fit_rates), sigma=unp.std_devs(fit_rates), absolute_sigma=True)

figure('forma')
clf()
errorbar(angles, unp.nominal_values(rates), yerr=unp.std_devs(rates), fmt=',k', capsize=2)
ylabel('tasso nei canali %d-%d [$s^{-1}$]\nfondo sottratto = %s $s^{-1}$' % (cut[0], cut[1], format(rate_fondo)))
xlabel('angolo [°]')
title('forma del fascio')
yscale('log')
grid()
show()
