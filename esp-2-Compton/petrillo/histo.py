from pylab import *
import numpy as np
from matplotlib import pyplot as plt
import sys

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
    dup_edges = np.empty(2 * len(edges))
    dup_edges[2 * np.arange(len(edges))] = edges
    dup_edges[2 * np.arange(len(edges)) + 1] = edges
    dup_counts = np.zeros(2 * len(edges))
    dup_counts[2 * np.arange(len(edges) - 1) + 1] = counts
    dup_counts[2 * np.arange(len(edges) - 1) + 2] = counts
    if ax is None:
        ax = plt.gca()
    return ax.plot(dup_edges, dup_counts, **kwargs)
    
def partial_sum(a, n):
    out = np.zeros(len(a) // n)
    for i in range(n):
        out += a[i::n][:len(out)]
    return out

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

# eventually merge bins
counts = partial_sum(counts, cut)

# plot histogram
figure('histo')
clf()
bar_line(arange(len(counts) + 1) - 0.5, counts, color='black', linewidth=.25)
if logscale:
    yscale('symlog', linthreshy=1, linscaley=1/5, subsy=[2, 3, 4, 5, 6, 7, 8, 9])
xlabel('canale ADC')
ylabel('conteggio')
xticks(arange(9) * 1000, ['%dk' % i for i in range(9)])
minorticks_on()
show()
