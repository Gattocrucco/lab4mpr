import tagload
import matplotlib.pyplot as plt
import numpy as np
import uncertainties as un
from uncertainties import unumpy as unp
import lab4

labels = ['dati 15 marzo', 'dati 19 marzo']
colors = ['lightgray'    , 'black'        ]
fmts   = ['^'            , '.'            ]

fig = plt.figure('soglia')
fig.clf()
fig.set_tight_layout(True)

ax = fig.add_subplot(111)

for label, color, fmt in zip(labels, colors, fmts):
    # dati 15 marzo
    if '15' in label:
        data, tags = tagload.tagload('../dati/0315-soglia.txt')

        sorted_thresh = tags['soglia']
        thresh = np.unique(tags['soglia'])
        counts = np.empty(thresh.shape)
        times = np.zeros(thresh.shape, dtype=object)

        for i in range(len(thresh)):
            this_data = data[sorted_thresh == thresh[i]]
            counts[i] = np.sum(this_data)
            for j in range(len(this_data)):
                times[i] += un.ufloat(10000, 0.5) * 1e-3
        counts = unp.uarray(counts, np.sqrt(counts))
        rates = counts / times

    # dati 19 marzo
    elif '19' in label:
        thresh, counts, clock = np.loadtxt('../dati/0319-soglia.txt', unpack=True)
        counts = unp.uarray(counts, np.sqrt(counts))
        times = unp.uarray(clock, 0.5) * 1e-3
        rates = counts / times

    lab4.errorbar(thresh, rates, ax=ax, fmt=fmt, color=color, label=label, markersize=3)

ax.set_ylabel('rate [s$^{-1}$]')
ax.set_xlabel('soglia discriminatore')
ax.grid(linestyle=':')
ax.legend(loc='best', fontsize='small')

fig.show()
