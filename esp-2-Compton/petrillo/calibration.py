import numpy as np
import matplotlib.pyplot as plt

nom_energy, adc_energy, adc_energy_unc, adc_sigma, adc_sigma_unc, adc_energy_sigma_cov = np.loadtxt('cippa.txt', unpack=True)

fig = plt.figure('calibration')
fig.clf()

ax_cal = fig.add_subplot(211)
ax_res = fig.add_subplot(212)

ax_cal.errorbar(nom_energy, adc_energy, yerr=adc_energy_unc, fmt=',k', capsize=2)
ax_cal.set_ylabel('fitted peak center [digit]')

ax_res.errorbar(nom_energy, adc_sigma, yerr=adc_sigma_unc, fmt=',k', capsize=2)
ax_res.set_ylabel('fitted peak sigma [digit]')
ax_res.set_xlabel('nominal energy [MeV]')

fig.show()

def energy_sigma(E):
    """
    E = energy [MeV]
    energy resolution of NaI + PMT
    TO BE MODIFIED
    also: find the reference
    """
    return (2.27 + 7.28 * E ** -0.29 - 2.41 * E ** 0.21) * E / (100 * 2.35)

def energy_calibration(E):
    """
    E = energy [MeV]
    TO BE MODIFIED
    returns non-calibrated energy
    """
    return E
