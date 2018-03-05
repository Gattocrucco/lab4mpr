from pylab import *
import numpy as np
from matplotlib import pyplot as plt
import argparse
import lab
import logtonpy

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

def histo(datasets, ax=None, logscale=False, cut=1, **kw):
    """
    Plot histograms(s) of dataset(s).
    
    Parameters
    ----------
    datasets : dictionary
        <label>=<dataset>.
    ax : subplot or None
        If None, create new figure.
    logscale : bool, defaults to False
        If True, use vertical log scale.
    cut : integer >= 1
        Rebinning.
    
    Returns
    -------
    Subplot object, either given or created.
    """
    if ax is None:
        fig = plt.figure('histo')
        fig.clf()
        ax = fig.add_subplot(111)
        show = True
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
        bar_line(arange(0, 2**13 + 1, cut), counts, label='%s, N=%s' % (label, lab.num2si(np.sum(counts), format='%.3g')), ax=ax, **kw)
        if logscale:
            ax.set_yscale('symlog', linthreshy=1, linscaley=1/5, subsy=[2, 3, 4, 5, 6, 7, 8, 9])
    ax.set_xlabel('canale ADC')
    ax.set_ylabel('conteggio')
    ax.legend(loc=1, fontsize='small')
    ax.minorticks_on()
    ax.grid()
    if show:
        fig.show()
    
    return ax

if __name__ == '__main__':
    # read command line
    parser = argparse.ArgumentParser(description='Plot histograms of ADC dataset(s).')
    parser.add_argument('-l', '--log', action='store_true', default=False, help='use logarithmic vertical scale')
    parser.add_argument('-r', '--rebin', default=1, type=int, help='rebin merging groups of REBIN bins')
    parser.add_argument(metavar='file', nargs='+', help='.dat or .log/.npy file', dest='filenames')
    args = parser.parse_args()

    # load file
    datasets = {}
    for filename in args.filenames:
        print('loading {}...'.format(filename))
        if filename.endswith('.dat'):
            counts = np.loadtxt(filename, unpack=True, dtype='int32')
        elif filename.endswith('.log') or filename.endswith('.npy'):
            if filename.endswith('.log'):
                samples = logtonpy.logtonpy(filename)
            else:
                samples = np.load(filename)
    
            print('converting samples to counts...')
            counts = np.bincount(samples, minlength=2**13)
        else:
            raise ValueError('filename %s is neither .dat, .log or .npy' % (filename,))
        datasets[filename] = counts

    # plot histogram
    figure(1)
    kw = dict(logscale=args.log, cut=args.rebin, linewidth=min(.25 * args.rebin, 1))
    if len(datasets) == 1:
        kw.update(color='black')
    histo(datasets, **kw)
    show()
