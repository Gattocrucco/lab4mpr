import tagload
import matplotlib.pyplot as plt
import numpy as np

data, tags = tagload.tagload('../dati/0315-soglia.txt')

sorted_thresh = tags['soglia']
thresh = np.unique(tags['soglia'])
counts = np.empty(thresh.shape)

for i in range(len(thresh)):
    counts[i] = np.sum(data[sorted_thresh == thresh[i]])

fig = plt.figure('soglia')
fig.clf()
fig.set_tight_layout(True)

ax = fig.add_subplot(111)
ax.errorbar(thresh, counts, yerr=np.sqrt(counts), fmt='.k')
ax.set_ylabel('conteggio')
ax.set_xlabel('soglia discriminatore')
ax.grid(linestyle=':')

fig.show()
