import numpy as np
import calibration_single
import lab4
import lab
from uncertainties import unumpy

## As module:
# Use calibration_multiple() to get the parameters
# of the calibration line.
## As script:
# Run calibration_multiple() with the label specified on the
# command line, and plot the calibration.

def calibration_multiple(label, axcal=None, axres=None, **kw):
    """
    For the calibration labeled <label>, run calibration_single
    on each entry and fit a straight line to the means of the peaks.
    Optionally plot.
    
    Returns
    -------
    outputs : list
        List containing, for each channel (in order 1 2 3),
        a tuple of ufloats (m, q) which are parameters of the
        line fitted.
    """
    records = calibration_single.load_records()[label]
    out = []
    for record in records:
        out.append(calibration_single.calibration_single(*record))
    out = np.array(out, dtype=object).T

    energy = out[0]
    adc_energy = out[1]
    adc_sigma = out[2]
    channels = np.array(records, dtype=object)[:, 1]
    
    outputs = []
    for channel in np.sort(np.unique(channels)):
        # fit
        cut = channels == channel
        function = lambda x, m, q: m * x + q
        out = lab.fit_curve(function, energy[cut], adc_energy[cut], p0=[1, 1], tags='cal', **kw)
        outputs.append(out.upar)
        
        # plot
        if not (axcal is None):
            line, = axcal.plot(energy[cut], function(energy[cut], *out.par))
            color = line.get_color()
            lab4.errorbar(energy[cut], adc_energy[cut], ax=axcal, fmt='.', label='ch{:d}'.format(int(channel)), color=color)
        if not (axres is None):
            lab4.errorbar(energy[cut], adc_sigma[cut], ax=axres, fmt='.')
    
    return outputs

if __name__ == '__main__':
    from matplotlib import pyplot as plt
    import sys
    
    label = sys.argv[1]

    fig = plt.figure('calibration_multiple')
    fig.clf()
    fig.set_tight_layout(True)
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)
    
    out = calibration_multiple(label, axcal=ax1, axres=ax2, print_info=1)

    ax2.set_xlabel('energia nominale [keV]')
    ax1.set_ylabel('media del picco [digit]')
    ax2.set_ylabel('sigma del picco [digit]')
    ax1.legend(loc='best')
    ax1.grid(linestyle=':')
    ax2.grid(linestyle=':')

    fig.show()
    