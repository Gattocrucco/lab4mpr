import numpy as np
import matplotlib.pyplot as plt
import lab
import uncertainties as un

print('calibration...')

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

ms, qs, ams = {}, {}, {}

for date in unique_dates:
    data_date = data[:, dates == date]
    labels_date = labels[dates == date]
    nom_energy, adc_energy, adc_sigma, adc_energy_unc, adc_sigma_unc, adc_energy_sigma_cov = data_date
    
    # offset = len(np.unique(labels_date)) > 1
    # offset = True
    # par, cov = lab.fit_linear(nom_energy, adc_energy, dy=adc_energy_unc, offset=offset, absolute_sigma=False)
    # chisq = np.sum((adc_energy - (nom_energy * par[0] + par[1]))**2 / adc_energy_unc**2)
    # chisq_dof = len(adc_energy) - (2 if offset else 1)
    # if __name__ == '__main__':
    #     print('{}: m = {}, q = {}, chi2/ndof = {:.1f}/{}'.format(date, *lab.xe(par, np.sqrt(np.diag(cov))), chisq, chisq_dof))
    # scale_factor_cal = chisq / chisq_dof
    out = lab.fit_curve(lambda x, m, q: m * x + q, nom_energy, adc_energy, dy=adc_energy_unc, absolute_sigma=False, p0=[4000, 10])
    par, cov = out.par, out.cov
    if __name__ == '__main__':
        print('{}: m = {}, q = {}, chi2/ndof = {:.1f}/{}'.format(date, *lab.xe(par, np.sqrt(np.diag(cov))), out.chisq, out.chisq_dof))
    scale_factor_cal = out.chisq / out.chisq_dof
    ms[date], qs[date] = un.correlated_values(par, cov, tags=['calibration'] * 2)
    
    if __name__ == '__main__':
        ec = ax_cal.errorbar(nom_energy, adc_energy, yerr=adc_energy_unc, fmt='.', label=date)
        color = ec.lines[0].get_color()
        fx = np.linspace(np.min(nom_energy), np.max(nom_energy), 500)
        ax_cal.plot(fx, par[0] * fx + par[1], '-', color=color)
        ax_cal.set_ylabel('peak center [digit]')
        ax_cal.legend()
        ax_cal.grid()

    out = lab.fit_curve(energy_sigma_fit, nom_energy, adc_sigma, dy=adc_sigma_unc, p0=1, absolute_sigma=False)
    if __name__ == '__main__':
        print('       ampl = {}, chi2/ndof = {:.1f}/{}'.format(lab.xe(out.par[0], np.sqrt(out.cov[0,0])), out.chisq, out.chisq_dof))
    scale_factor_res = out.chisq / out.chisq_dof
    ams[date] = un.ufloat(out.par[0], np.sqrt(out.cov[0,0]), tag='resolution')
    
    if __name__ == '__main__':
        ax_res.errorbar(nom_energy, adc_sigma, yerr=adc_sigma_unc, fmt='.', color=color)
        ax_res.plot(fx, energy_sigma_fit(fx, *out.par), '-', color=color)
        ax_res.set_ylabel('peak sigma [digit]')
        ax_res.set_xlabel('nominal energy [MeV]')
        ax_res.grid()

if __name__ == '__main__':
    fig.show()

def energy_sigma(date='22feb', unc=False):
    a = ams[date]
    if not unc:
        a = a.n
    def fun(E):
        return energy_sigma_fit(E, a)
    return fun

def energy_calibration(date='22feb', unc=False):
    m = ms[date]
    q = qs[date]
    if not unc:
        m = m.n
        q = q.n
    def fun(E):
        return m * E + q
    return fun

def energy_inverse_calibration(date='22feb', unc=False):
    m = ms[date]
    q = qs[date]
    if not unc:
        m = m.n
        q = q.n
    def fun(digit):
        return (digit - q) / m
    return fun
