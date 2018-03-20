import numpy as np
import matplotlib.pyplot as plt
import lab4
from uncertainties import unumpy as unp

# load data
pres, unc_pres, count, clock = np.loadtxt('../dati/0320-pressione.txt', unpack=True)
pres_text = np.loadtxt('../dati/0320-pressione.txt', usecols=(0,), dtype=str)

# compute rates
count = unp.uarray(count, np.sqrt(count))
time = unp.uarray(clock, 0.5) * 1e-3
rate = count / time
pres = unp.uarray(pres, unc_pres)

# load spectra
mode = np.empty(len(pres_text))
mode_err = np.empty((2, len(pres_text)))
for i in range(len(pres_text)):
    file_name = '../de0_data/0320pres{}.dat'.format(pres_text[i])
    print('processing {}...'.format(file_name))
    t, ch1, ch2 = np.loadtxt(file_name, unpack=True)
    out = lab4.credible_interval(ch1, cl=0.9)
    mode[i] = out[0]
    mode_err[0, i] = out[0] - out[1]
    mode_err[1, i] = out[2] - out[0]

# plot
fig = plt.figure('pressione')
fig.clf()
fig.set_tight_layout(True)

ax1 = fig.add_subplot(211)
lab4.errorbar(pres, rate, fmt='.k', ax=ax1)
ax1.set_xscale('log')
ax1.set_ylabel('rate [s$^{-1}$]')
ax1.grid(linestyle=':')

ax2 = fig.add_subplot(212)
lab4.errorbar(pres, mode, yerr=mode_err, fmt='.k', ax=ax2)
ax2.set_xscale('log')
ax2.set_ylabel('moda (90 % CR) [digit]')
ax2.set_xlabel('pressione [mbar]')
ax2.grid(linestyle=':')

fig.show()
