import numpy as np
import matplotlib.pyplot as plt
import sys

filenames = sys.argv[1:]

fig = plt.figure('histo')
fig.clf()
fig.set_tight_layout(True)

def unroll_time(t):
    # non è ancora del tutto buona perché non tiene conto del fatto
    # che ci sono più dati a tempo fissato, però per i nostri scopi
    # va bene.
    tmax = 6553.5
    # preso da max(t)
    # bisogna sommare 65535 e non 65536 perché min(t) == 0.1
    diff = np.diff(t)
    cycles = np.concatenate([[0], np.cumsum(diff < 0)])
    return t + tmax * cycles

if len(filenames) > 1:
    datasets = []
    for filename in filenames:
        print('loading %s...' % (filename,))
        t, ch1, ch2 = np.loadtxt(filename, unpack=True)
        datasets.append(ch1)
    ax = fig.add_subplot(111)
    nbinspow = min(int(np.ceil(np.log2(np.sqrt(max([len(ds) for ds in datasets]))))), 12)
    edges = np.arange(2 ** 12 + 1)[::2 ** (12 - nbinspow)] - 0.5
    ax.hist(datasets, bins=edges, density=True, histtype='step', label=filenames)
    ax.legend(loc='best', fontsize='small')
    ax.set_xlabel('canale ADC')
    ax.set_ylabel('densita')

elif len(filenames) == 1:
    filename = filenames[0]
    print('loading %s...' % (filename,))
    rolled_t, ch1, ch2 = np.loadtxt(filename, unpack=True)
    t = unroll_time(rolled_t)
    ax1 = fig.add_subplot(211)
    nbinspow = min(int(np.ceil(np.log2(np.sqrt(len(ch1))))), 12)
    edges = np.arange(2 ** 12 + 1)[::2 ** (12 - nbinspow)] - 0.5
    ax1.hist(ch1, bins=edges, histtype='step', label=filename)
    ax1.legend(loc='best', fontsize='small')
    ax1.set_ylabel('conteggio')
    ax1.set_xlabel('canale ADC')
    ax2 = fig.add_subplot(212)
    ax2.plot(t, ch1, '.', markersize=2)
    ax2.set_xlabel('tempo')
    ax2.set_ylabel('canale ADC')

else:
    print('no filenames specified.')

fig.show()

