import numpy as np
import matplotlib.pyplot as plt
import lab
import uncertainties as un
from scipy import optimize

print('calibration...')

filename = '../bob/cal/cal.txt'

# empirical model for resolution
def energy_sigma_fit(E, ampl):
    return ampl * (2.27 + 7.28 * E ** -0.29 - 2.41 * E ** 0.21) * E / (100 * 2.35)

# load data
data = np.loadtxt(filename, unpack=True, usecols=(2,3,4,5,6,7,8))
dates, labels = np.loadtxt(filename, unpack=True, usecols=(0,1), dtype=str)

# set up figure
if __name__ == '__main__':
    fig = plt.figure('calibration')
    fig.clf()
    
    ax_cal = fig.add_subplot(211)
    ax_res = fig.add_subplot(212)

# calibrations are identified by date
unique_dates = np.unique(dates)

# function to fit all calibrations simultaneously
def fit_fun(x, *p):
    # x is actually ignored
    y = np.empty(len(dates))
    n = 0
    for i in range(len(unique_dates)):
        date = unique_dates[i]
        nom_energy = data[0, dates == date]
        
        m = p[2 * i]
        q = p[2 * i + 1]
        
        y[n:n + len(nom_energy)] = m * nom_energy + q
        n += len(nom_energy)
    return y

# data containers for the fit
y = []
covy = np.zeros((len(dates), len(dates)))

# fill data containers, with the same ordering of fit_fun
for i in range(len(unique_dates)):
    date = unique_dates[i]
    data_date = data[:, dates == date]
    labels_date = labels[dates == date]
    nom_energy, adc_energy, _, adc_energy_unc, _, _, co60_cov = data_date
    
    for j in range(len(y), len(y) + len(adc_energy)):
        covy[j,j] = adc_energy_unc[j - len(y)] ** 2
    idxs = np.arange(len(y), len(y) + len(adc_energy), dtype=int)[labels_date == 'Co60']
    covy[idxs[0], idxs[1]] = co60_cov[co60_cov != 0]
    covy[idxs[1], idxs[0]] = covy[idxs[0], idxs[1]]
    
    y += list(adc_energy)
    
y = np.array(y)

# run fit
p0 = [5000, 10] * len(unique_dates)
par, cov = optimize.curve_fit(fit_fun, None, y, sigma=covy, p0=p0, absolute_sigma=False)

# plot calibrations and fit resolutions separately
# also fills results containers for interface
ms, qs, ams = {}, {}, {}
for i in range(len(unique_dates)):
    date = unique_dates[i]
    data_date = data[:, dates == date]
    labels_date = labels[dates == date]
    nom_energy, adc_energy, adc_sigma, adc_energy_unc, adc_sigma_unc, _, _ = data_date
        
    m = par[2 * i]
    q = par[2 * i + 1]
    c = cov[np.ix_((2*i,2*i+1),(2*i,2*i+1))]

    if __name__ == '__main__':
        ec = ax_cal.errorbar(nom_energy, adc_energy, yerr=adc_energy_unc, fmt='.', label=date)
        color = ec.lines[0].get_color()
        fx = np.linspace(np.min(nom_energy), np.max(nom_energy), 500)
        ax_cal.plot(fx, m * fx + q, '-', color=color)
        print('{}: m = {}, q = {}'.format(date, *lab.xe([m,q], np.sqrt(np.diag(c)))))
    
    # force different calibrations to be uncorrelated
    # (actually true, but the fit is numerical)
    m, q = un.correlated_values([m, q], c, tags=['calibration'] * 2)
    ms[date] = m
    qs[date] = q
    
    out = lab.fit_curve(energy_sigma_fit, nom_energy, adc_sigma, dy=adc_sigma_unc, p0=1, absolute_sigma=False)
    if __name__ == '__main__':
        print('       ampl = {}, chi2/ndof = {:.1f}/{}'.format(lab.xe(out.par[0], np.sqrt(out.cov[0,0])), out.chisq, out.chisq_dof))
    ams[date] = un.ufloat(out.par[0], np.sqrt(out.cov[0,0]), tag='resolution')
    
    if __name__ == '__main__':
        ax_res.errorbar(nom_energy, adc_sigma, yerr=adc_sigma_unc, fmt='.', color=color)
        ax_res.plot(fx, energy_sigma_fit(fx, *out.par), '-', color=color)

if __name__ == '__main__':
    ax_cal.set_ylabel('peak center [digit]')
    ax_cal.grid()
    ax_cal.legend()
    
    ax_res.set_ylabel('peak sigma [digit]')
    ax_res.set_xlabel('nominal energy [MeV]')
    ax_res.grid()
    
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
