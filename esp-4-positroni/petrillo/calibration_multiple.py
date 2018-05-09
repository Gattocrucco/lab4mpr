import numpy as np
import calibration_single
from matplotlib import pyplot as plt
import lab4
import lab
from uncertainties import unumpy

# arguments
label = '0504_2'

records = calibration_single.load_records()[label]
out = []
for record in records:
    out.append(calibration_single.calibration_single(*record))
out = np.array(out, dtype=object).T

energy = out[0]
adc_energy = out[1]
adc_sigma = out[2]
channels = np.array(records, dtype=object)[:, 1]

fig = plt.figure('calibration_multiple')
fig.clf()
fig.set_tight_layout(True)
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)

for channel in np.unique(channels):
    cut = channels == channel
    function = lambda x, m, q: m * x + q
    out = lab.fit_curve(function, energy[cut], adc_energy[cut], p0=[1, 1], print_info=1)
    line, = ax1.plot(energy[cut], function(energy[cut], *out.par))
    color = line.get_color()
    lab4.errorbar(energy[cut], adc_energy[cut], ax=ax1, fmt='.', label='ch{:d}'.format(int(channel)), color=color)
    lab4.errorbar(energy[cut], adc_sigma[cut], ax=ax2, fmt='.', color=color)

ax2.set_xlabel('energia nominale [keV]')
ax1.set_ylabel('media del picco [digit]')
ax2.set_ylabel('sigma del picco [digit]')
ax1.legend(loc='best')

fig.show()
