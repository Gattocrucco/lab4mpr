import lab
import lab4
import numpy as np
import sys
import os
from matplotlib import pyplot as plt

filenames = sys.argv[1:]

# check that files exist
for filename in filenames:
    if not os.path.exists(filename):
        raise RuntimeError('File `{}` does not exist.'.format(filename))

fig = plt.figure('histofit')
fig.clf()

ax = fig.add_subplot(111)

def gauss(x, peak, mean, sigma):
    return peak * np.exp(-(x - mean) ** 2 / sigma ** 2)

outputs = []

for filename in filenames:
    print('\n__________{}__________\n'.format(filename))
    
    # load file
    ch1, ch2, ch3, tr1, tr2, tr3, c2, c3, ts = lab4.loadtxt(filename,  unpack=True,  usecols=(0, 1, 2, 4, 5, 6, 8, 9, 12))
    
    # histogram
    bins = np.arange(0, 1200 // 8) * 8
    counts, edges = np.histogram(ch1[c2 > 500], bins=bins)
    
    # fit
    x = (edges[1:] + edges[:-1]) / 2
    p0 = [1] * 3
    argmax = np.argmax(counts)
    # initial parameters
    p0[0] = counts[argmax] # peak
    p0[1] = x[argmax] # mean
    p0[2] = 50 # sigma
    cut = (counts > counts[argmax] / 3) & (x < 1024)
    if np.sum(cut) > 1:
        out = lab.fit_curve(gauss, x[cut], counts[cut], p0=p0, dy=np.sqrt(counts)[cut], print_info=1)
        outputs.append(out)
    else:
        outputs.append(None)
    
    # plot
    line, = lab4.bar(edges, counts, ax=ax, label=filename)
    if not outputs[-1] is None:
        color = line.get_color()
        xspace = np.linspace(np.min(x[cut]), np.max(x[cut]), 100)
        ax.plot(xspace, gauss(xspace, *out.par), '--', color=color)

ax.legend(loc='best')
ax.set_xlabel('canale ADC')
ax.set_ylabel('conteggio / 8 canali')

for i in range(len(outputs)):
    if outputs[i] is None:
        continue
    mean = outputs[i].upar[1]
    sigma = outputs[i].upar[2]
    rel = abs(sigma) / mean
    print(('{:%ds}: {:7P}' % max(map(len, filenames))).format(filenames[i], rel))

fig.show()
