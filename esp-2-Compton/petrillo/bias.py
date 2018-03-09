import mc9
from matplotlib import pyplot as plt
import histo
import numpy as np
import lab
import calibration
from uncertainties import umath
import uncertainties as un

def bias(energy=1.33, theta_0=10, calib_date='22feb', N=100000):
    p, _, wp, _ = mc9.mc_cached(energy, theta_0=un.nominal_value(theta_0), seed=32, N=100000, beam_center=0, m_e=0.511)

    mu = np.average(p, weights=wp)
    mu_unc = np.std(p) / np.sqrt(len(p))
    mu = un.ufloat(mu, mu_unc, tag='mc')
    mu_cal = calibration.energy_inverse_calibration(date=calib_date, unc=True)(mu)

    m_fit = (1 - umath.cos(umath.radians(theta_0))) / (1 / mu_cal - 1 / energy)
    m_teo = 0.511

    bias = m_fit - m_teo
    
    return bias
