from pylab import *
import numpy as np
from matplotlib import pyplot as plt
import sys
import lab

cut = 1

# utilizzo:
# histo.py filename.dat       # --> file istogramma
# histo.py filename.npy/.log  # --> file con tutti i campioni, istogramma da calcolare
# histo.py filename.xxx log   # --> scala verticale logaritmica

def bar_line(edges, counts, ax=None, **kwargs):
    """
    Draw histogram with solid skyline.
    
    Parameters
    ----------
    edges, counts :
        As returned e.g. by numpy.histogram.
    ax : None or matplotlib axis
        If None use current axis, else axis given.
    
    Keyword arguments
    -----------------
    Passed to ax.plot.
    
    Returns
    -------
    Return value from ax.plot.
    """
    dup_edges = np.concatenate([[edges[0]], edges, [edges[-1]]])
    dup_counts = np.concatenate([[0], [counts[0]], counts, [0]])
    if ax is None:
        ax = plt.gca()
    kwargs.update(drawstyle='steps')
    return ax.plot(dup_edges, dup_counts, **kwargs)
    
def partial_sum(a, n):
    out = np.zeros(len(a) // n)
    for i in range(n):
        out += a[i::n][:len(out)]
    return out

def histo(datasets, figname='histo', logscale=False, cut=1, **kw):
    """
    Plot histograms(s) of dataset(s).
    
    Parameters
    ----------
    datasets : dictionary
        <label>=<data>.
    figname : string, defaults to 'histo'
        Name of the figure window.
    logscale : bool, defaults to False
        If True, use vertical log scale.
    """
    figure(figname)
    clf()
    for label in datasets.keys():
        data = datasets[label]
        if data.dtype == np.dtype('uint16'):
            counts = np.bincount(data, minlength=2**13)
        elif len(data) == 2**13:
            counts = data
        else:
            raise ValueError('data for label %s not recognised as samples or histogram' % label)
        if cut > 1:
            counts = partial_sum(counts, cut)
        bar_line(arange(len(counts) + 1) - 0.5, counts, label='%s, N=%s' % (label, lab.num2si(np.sum(counts), format='%.3g')), **kw)
        if logscale:
            yscale('symlog', linthreshy=1, linscaley=1/5, subsy=[2, 3, 4, 5, 6, 7, 8, 9])
    xlabel('canale ADC')
    ylabel('conteggio')
    # xticks(arange(9) * 1000 / cut, ['%dk' % i for i in range(9)])
    legend(loc=1, fontsize='small')
    minorticks_on()
    grid()
    show()

if __name__ == '__main__':
    # read command line
    filename = sys.argv[1]
    logscale = 'log' in sys.argv[2:]

    # load file
    print('loading file...')
    if filename.endswith('.dat'):
        counts = np.loadtxt(filename, unpack=True, dtype='int32')
    elif filename.endswith('.log') or filename.endswith('.npy'):
        if filename.endswith('.log'):
            samples = np.loadtxt(filename, unpack=True, converters={0: lambda s: int(s, base=16)}, dtype='uint16')
        else:
            samples = np.load(filename)
    
        print('converting samples to counts...')
        counts = np.bincount(samples, minlength=2**13)
    else:
        raise ValueError('filename %s is neither .dat, .log or .npy' % (filename,))

    # plot histogram
    histo({filename: counts}, logscale=logscale, cut=cut, linewidth=.25, color='black')
