import mc9
from matplotlib import pyplot as plt
import histo
import numpy as np
import lab
import calibration
from uncertainties import umath
import uncertainties as un
import empirical

def bias(energy=1.33, theta_0=10, calib_date='22feb', N=100000):
    p, _, wp, _ = mc9.mc_cached(energy, theta_0=theta_0, seed=32, N=N, beam_center=0, m_e=0.511, date=calib_date)

    mu = np.average(p, weights=wp)
    mu_unc = np.std(p) / np.sqrt(len(p))
    mu = un.ufloat(mu, mu_unc)
    mu_cal = calibration.energy_inverse_calibration(date=calib_date, unc=False)(mu)

    m_fit = (1 - np.cos(np.radians(theta_0))) / (1 / mu_cal - 1 / energy)

    bias = m_fit - 0.511
    
    return bias

def gauss(x, N, mu, sigma):
    return N / (np.sqrt(2 * np.pi) * sigma) * np.exp(-1/2 * (x - mu)**2 / sigma**2)

def fit_fun(x, N1, mu1, sigma1, N2, mu2, sigma2):
    return gauss(x, N1, mu1, sigma1) + gauss(x, N2, mu2, sigma2)

def bias_double(energy1=1.33, energy2=1.17, theta_0=45, calib_date='22feb', N=100000):
    p1, _, wp1, _ = mc9.mc_cached(energy1, theta_0=theta_0, seed=64, N=N, beam_center=0, m_e=0.511, date=calib_date)
    p2, _, wp2, _ = mc9.mc_cached(energy2, theta_0=theta_0, seed=128, N=N, beam_center=0, m_e=0.511, date=calib_date)
    
    c, e, dc = empirical.histogram(np.concatenate([p1, p2]), bins=int(np.sqrt(2 * N)), weights=np.concatenate([wp1, wp2]))
    
    p0 = [np.sum(wp1), np.mean(p1), np.std(p1), np.sum(wp2), np.mean(p2), np.std(p2)]
    out = lab.fit_curve(fit_fun, e[:-1] + (e[1] - e[0]) / 2, c, dy=dc, p0=p0)
    
    mu_cal1 = calibration.energy_inverse_calibration(date=calib_date, unc=False)(out.upar[1])
    m_fit1 = (1 - np.cos(np.radians(theta_0))) / (1 / mu_cal1 - 1 / energy1)

    mu_cal2 = calibration.energy_inverse_calibration(date=calib_date, unc=False)(out.upar[4])
    m_fit2 = (1 - np.cos(np.radians(theta_0))) / (1 / mu_cal2 - 1 / energy2)
    
    return m_fit1 - 0.511, m_fit2 - 0.511
