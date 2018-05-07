import numpy as np
import lab
import lab4

# This script can be used as module or executed.
## As module:
# function calibration_single():
# Fit and/or plot a histogram containing a reference peak.
# function load_records():
# Load the file ../dati/calibration_single.txt as a dictionary.
## As script:
# Specify an ADC file on the command line
# and run calibration_single() on that file,
# using the first entry containing that file in ../dati/calibration_single.txt.

# dictionary mapping source label to energy in keV
sources = {
    'na': 1275,
    'co17': 1173,
    'co33': 1333,
    'cs': 662
}

def calibration_single(filename, channel, source, cut, ax=None, **kw):
    """
    Calibration fit of single peak.
    
    Parameters
    ----------
    filename : string
        ADC file.
    channel : integer
        Channel to read in ADC file.
    source : string
        Label corresponding to energy of peak to be fitted.
        One of na, co17, co33, cs.
    cut : interval [left, right]
        Interval of ADC channels to be fitted.
    ax : subplot or None
        If not None, plot data and fit.
    
    Keyword arguments
    -----------------
    Additional keyword arguments are passed to lab.fit_curve.
    
    Returns
    -------
    energy : number
        Nominal peak energy in keV.
    adc_energy : ufloat
        Value of the energy in the ADC scale.
    adc_sigma : ufloat
        Sigma of the peak in the ADC units. adc_energy and adc_sigma
        are correlated variables.
    """
    # load data
    ch1, ch2, ch3, tr1, tr2, tr3, c2, c3, ts = lab4.loadtxt(filename, unpack=True, usecols=(0, 1, 2, 4, 5, 6, 8, 9, 12))
    samples = np.array([ch1, ch2, ch3][channel - 1][[tr1, tr2, tr3][channel - 1] > 500], dtype=int)
    hist = np.bincount(samples)
    bins = np.arange(len(hist) + 1)
    # missing_codes = bins[(bins // 4) % 2 == 0]

    # fit
    # prepare data
    x = (bins[1:] + bins[:-1]) / 2
    y = hist
    dy = np.sqrt(y)
    # cut
    cut = (y > 0) & (x >= cut[0]) & (x <= cut[1])
    if np.sum(cut) > 3:
        x, y, dy = np.array([x, y, dy])[:, cut]
        # fit function
        def gauss(x, peak, mean, sigma):
            return peak * np.exp(-(x - mean)**2 / sigma**2)
        def fit_fun(x, *par):
            return gauss(x, *par) + gauss(x - 4, *par)
        # initial parameters
        p0 = (np.max(y) / 2, np.mean(x), (np.max(x) - np.min(x)) / 2)
        out = lab.fit_curve(fit_fun, x, y, dy=dy, p0=p0, **kw)
    else:
        out = None

    # plot
    if not (ax is None):
        # data
        lab4.bar(bins, hist, ax=ax, label=filename.split('/')[-1])
        # fit
        if not (out is None):
            xspace = np.linspace(np.min(x), np.max(x), 100)
            if out.success:
                ax.plot(xspace, fit_fun(xspace, *out.par), '-k', label='fit ' + source)
            else:
                ax.plot(xspace, fit_fun(xspace, *p0), '-r', label='p0 (fit fallito)')
        # decorations
        ax.set_xlabel('canale ADC')
        ax.set_ylabel('conteggio')
        ax.legend(loc='best')
    
    return (sources[source],) + ((None, None) if out is None else tuple(out.upar[1:]))

def load_records(file='../dati/calibration_single.txt', prepath='../DAQ'):
    """
    Load a file containing labels identifying calibration
    and the inputs to the function calibration_single.
    
    Parameters
    ----------
    file : string
        File to load.
    prepath : string
        Location prefixed to filenames contained in <file>.
    
    Returns
    -------
    A dictionary where the keys are the labels identifyng
    the calibrations and the values are lists of tuples
    where the tuples are argument lists for calibration_single.
    """
    label, filename, channel, source, cut = lab4.loadtxt(file, unpack=True, dtype=str)
    channel = np.array(channel, dtype=int)
    cut = list(map(eval, cut))
    unique_label = np.unique(label)
    dict_by_label = {}
    for lbl in unique_label:
        idxs = np.arange(len(label), dtype=int)[label == lbl]
        dict_by_label[lbl] = [(prepath + '/' + filename[i], channel[i], source[i], cut[i]) for i in idxs]
    return dict_by_label

if __name__ == '__main__':
    import sys
    import os
    from matplotlib import pyplot as plt
    
    # get filename from command line
    if len(sys.argv) < 2:
        raise ValueError('Specify filename on the command line.')
    filename = sys.argv[1]
    if not os.path.exists(filename):
        raise RuntimeError('File `{}` does not exist.'.format(filename))
    
    # find record in calibration file
    records = load_records()
    found = False
    for key in records.keys():
        for args in records[key]:
            if args[0] == filename:
                found = args
                break
        if found:
            break
    if not found:
        raise RuntimeError('File `{}` not found in records.')
    args = found
    
    # prepare figure
    fig = plt.figure('calibration-single')
    fig.clf()
    fig.set_tight_layout(True)
    ax = fig.add_subplot(111)
    
    # run calibration
    energy, adc_energy, adc_sigma = calibration_single(*args, ax=ax, print_info=1)
    
    fig.show()
    