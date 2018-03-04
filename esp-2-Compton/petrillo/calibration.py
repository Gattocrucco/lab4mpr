import numpy as np
import matplotlib.pyplot as plt
import lab

filename = '../bob/cal/cal.txt'

data = np.loadtxt(filename, unpack=True, usecols=(2,3,4,5,6,7))
dates, labels = np.loadtxt(filename, unpack=True, usecols=(0,1), dtype=str)

if __name__ == '__main__':
    fig = plt.figure('calibration')
    fig.clf()

    ax_cal = fig.add_subplot(211)
    ax_res = fig.add_subplot(212)

unique_dates = np.unique(dates)

def energy_sigma_fit(E, ampl):
    return ampl * (2.27 + 7.28 * E ** -0.29 - 2.41 * E ** 0.21) * E / (100 * 2.35)

for date in unique_dates:
    data_date = data[:, dates == date]
    labels_date = labels[dates == date]
    nom_energy, adc_energy, adc_sigma, adc_energy_unc, adc_sigma_unc, adc_energy_sigma_cov = data_date
    
    p0, _ = lab.fit_linear(nom_energy, adc_energy, dy=adc_energy_unc)
    out = lab.fit_curve(lambda x, m, q: m * x + q, nom_energy, adc_energy, dy=adc_energy_unc, p0=p0, absolute_sigma=False)
    print('{}: m = {}, q = {}, chi2/ndof = {:.1f}/{}'.format(date, *lab.xe(out.par, np.sqrt(np.diag(out.cov))), out.chisq, out.chisq_dof))
    scale_factor_cal = out.chisq / out.chisq_dof
    
    if __name__ == '__main__':
        ec = ax_cal.errorbar(nom_energy, adc_energy, yerr=adc_energy_unc, fmt='.', label=date)
        color = ec.lines[0].get_color()
        fx = np.linspace(np.min(nom_energy), np.max(nom_energy), 500)
        ax_cal.plot(fx, out.par[0] * fx + out.par[1], '-', color=color)
        ax_cal.set_ylabel('peak center [digit]')
        ax_cal.legend()

    out = lab.fit_curve(energy_sigma_fit, nom_energy, adc_sigma, dy=adc_sigma_unc, p0=1, absolute_sigma=False)
    print('       ampl = {}, chi2/ndof = {:.1f}/{}'.format(lab.xe(out.par[0], np.sqrt(out.cov[0,0])), out.chisq, out.chisq_dof))
    scale_factor_res = out.chisq / out.chisq_dof
    
    if __name__ == '__main__':
        ax_res.errorbar(nom_energy, adc_sigma, yerr=adc_sigma_unc, fmt='.', color=color)
        ax_res.plot(fx, energy_sigma_fit(fx, *out.par), '-', color=color)
        ax_res.set_ylabel('peak sigma [digit]')
        ax_res.set_xlabel('nominal energy [MeV]')

if __name__ == '__main__':
    fig.show()

def energy_calibration(E):
    """
    E = energy [MeV]
    TO BE MODIFIED
    returns non-calibrated energy
    """
    return E
