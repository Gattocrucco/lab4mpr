from pylab import *
import numpy as np
from matplotlib import pyplot as plt
import sys

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

counts = loadtxt(sys.argv[1], unpack=True)

logscale = 'log' in sys.argv[1:]

# cut = 1
#
# def partial_sum(a, n):
#     out = zeros(len(a) // n)
#     for i in range(n):
#         out += a[i::n][:len(out)]
#     return out
#
# counts = partial_sum(data, cut)

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
