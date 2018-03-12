import mc9
import histo
from mc9 import mc_cached
import numpy as np
import histo

N=100000
p, s, wp, ws = mc_cached(1.33, theta_0=15, N=N, seed=34, date="27feb")
p7, s7, wp7, ws7 = mc_cached(1.17, theta_0=15, N=N, seed=23, date="27feb")
ws/=6; ws7/=6

samples = np.concatenate([p, s, p7, s7])
weights = np.concatenate([wp, ws, wp7, ws7])

from matplotlib.pyplot import *
figure('sovrapposti', figsize=[5.37, 3.19]).set_tight_layout(True)
clf()

# dati

grezzi=np.loadtxt("../dati/histo-27feb-e15.dat")
totale = np.sum(grezzi[50:8050])
rebin = 16
counts = histo.partial_sum(grezzi, rebin)
edges = np.arange(8193)[::rebin]
histo.bar_line(edges, counts, color='gray', label='dati')

# montecarli
weights *= (totale / len(counts)) / (np.sum(weights) / np.floor(np.sqrt(len(samples))))
mc_counts, mc_edges = np.histogram(samples, bins=int(np.sqrt(len(samples))), weights=weights)
histo.bar_line(mc_edges, mc_counts, label='MC', color='black')
# wp *= totale/sum(wp) * np.sqrt(N) / len(counts)
# ws *= totale/sum(ws) * np.sqrt(N) / len(counts)
# hist(p, bins=int(np.sqrt(N)), weights=wp, histtype='step', label='1.33 MeV',color="orange")
# hist(s, bins=int(np.sqrt(N)), weights=ws, histtype='step', label='spalla 1.33 MeV',color="orange")

# wp7*=totale/sum(wp7) * np.sqrt(N) / len(counts)
# ws7*=totale/sum(ws7) * np.sqrt(N) / len(counts)
# hist(p7, bins=int(np.sqrt(N)), weights=wp7, histtype='step', label='1.17 MeV',color="blue")
# hist(s7, bins=int(np.sqrt(N)), weights=ws7, histtype='step', label='spalla 1.17 MeV',color="blue")

ymax = np.max(counts[50//rebin:8050//rebin])
ylim((-0.05 * ymax, 1.05 * ymax))
legend(loc='upper right')
xlabel('canale ADC [digit]')
ylabel('conteggio [(%d$\\cdot$digit)$^{-1}$]' % (rebin,))
show()