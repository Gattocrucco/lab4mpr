from pylab import *
import glob, scanf
import uncertainties as un
from uncertainties import unumpy as unp
from uncertainties import umath
import lab
import numpy as np
from scipy import integrate
import numba as nb

####### PARAMETERS #######

L = 40 # distanza del NaI
R = 2.54 # raggio del NaI

cut = (5500, 8040) # canali dell'ADC da usare
fitcut = (-4, 4) # intervallo di angoli da usare nel fit finale
angle_un = 0.1 # incertezza sulle misure di angoli

####### LOAD FILES #######

files = glob.glob('../dati/histo-20feb-ang*-?????.dat')
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
    string = file.replace('_', '-')
    angle, clock = scanf.scanf('%*s-ang%d-%u.dat', s=string)
    angle = float(angle)
    rate = un.ufloat(count, sqrt(count), tag='countang%.0f' % angle) / (clock * 1e-3) - rate_fondo
    rates.append(rate)
    angles.append(angle)
angles = array(angles)
rates = array(rates)

####### FIT #######

# fit segnale + fondo

def fit_signal(angle, top, center, sigma):
    return top * exp(-1/2 * (angle - center)**2 / sigma**2)
def fit_bkg(angle, top, center, sigma):
    return top * exp(-1/2 * (angle - center)**2 / sigma**2)
def fit_fun(angle, top, center, sigma, bkg_top, bkg_center, bkg_sigma):
    return fit_signal(angle, top, center, sigma) + fit_bkg(angle, bkg_top, bkg_center, bkg_sigma)
p0 = [
    max(unp.nominal_values(rates)), np.mean(angles), 2.5,
    1/10 * max(unp.nominal_values(rates)), np.mean(angles), 15
]
out_pre = lab.fit_curve(fit_fun, angles, unp.nominal_values(rates), dx=angle_un, dy=unp.std_devs(rates), absolute_sigma=True, p0=p0)

print('risultato del prefit  (signal, bkg) x (top, center, sigma):')
print(lab.format_par_cov(out_pre.par, out_pre.cov))

bool_cut = logical_and(fitcut[0] <= angles, angles <= fitcut[1])
fit_angles = angles[bool_cut]
fit_rates = rates[bool_cut]

# fit segnale con fondo fissato a quello ricavato dal fit precedente

def fit_signal_only(angle, *p):
    return fit_signal(angle, *p) + fit_bkg(angle, *out_pre.par[3:])

out = lab.fit_curve(fit_signal_only, fit_angles, unp.nominal_values(fit_rates), dx=angle_un, dy=unp.std_devs(fit_rates), absolute_sigma=True, p0=out_pre.par[:3])
par = out.par
cov = out.cov

print('risultato del fit (top, center, sigma):')
print(lab.format_par_cov(par, cov))
print('center = %s °' % lab.xe(par[1], sqrt(cov[1,1]), pm=lab.unicode_pm))
print('sigma = %s °' % lab.xe(par[2], sqrt(cov[2,2]), pm=lab.unicode_pm))
print('chi2/dof = %.1f/%d' % (out.chisq, out.chisq_dof))

# grafico

figure('forma')
clf()
errorbar(angles, unp.nominal_values(rates), xerr=angle_un, yerr=unp.std_devs(rates), fmt=',k', capsize=2, label='dati $-$ fondo c.\nfondo c. = %s $s^{-1}$' % ('{:P}'.format(rate_fondo),))
yscale('log')
x = xlim()
y = ylim()
fa = linspace(min(angles), max(angles), 500)
ffa = linspace(min(fit_angles), max(fit_angles), 500)
plot(fa, fit_fun(fa, *out_pre.par), '-.', color='gray', zorder=-10, label='fit segnale+fondo p.')
plot(fa, fit_signal(fa, *out_pre.par[:3]), ':', color='darkgray', zorder=-8, label='  segnale')
plot(fa, fit_bkg(fa, *out_pre.par[3:]), '--', color='darkgray', zorder=-9, label='  fondo p.')
plot(ffa, fit_signal_only(ffa, *par), '-', color='gray', zorder=-7, label='fit segnale (fondo fissato)')
xlim(x)
ylim(y)
ylabel('tasso nei canali %d-%d [$s^{-1}$]' % (cut[0], cut[1]))
xlabel('angolo [°]')
title('forma del fascio')
legend(loc=1, fontsize='small')
grid()
tight_layout()
show()

####### DECONVOLUTION #######

def integrand_mu0(theta):
    return np.sqrt(1 - (L/R)**2 * np.tan(theta)**2)

def integrand_mu1(theta):
    return integrand_mu0(theta) * theta

def integrand_mu2(theta):
    return integrand_mu0(theta) * theta**2

right = np.arctan(R / L)
left = -right

mu0, dmu0 = integrate.quad(integrand_mu0, left, right)
mu1, dmu1 = integrate.quad(integrand_mu1, left, right)
mu2, dmu2 = integrate.quad(integrand_mu2, left, right)

sigma2_nai = mu2/mu0 - (mu1/mu0)**2
sigma2_nai *= (180 / np.pi)**2

sigma_nai = un.ufloat(np.sqrt(sigma2_nai), np.sqrt(sigma2_nai) * 1.5 / 40, 'nai sigma')

sigma_beam = umath.sqrt(un.ufloat(par[2], np.sqrt(cov[2,2]), 'beam sigma')**2 - sigma_nai**2)

print('\nsigma nai = {:P} °\nsigma beam = {:P} °'.format(sigma_nai, sigma_beam))
