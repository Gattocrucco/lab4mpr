from matplotlib import pyplot as plt
from matplotlib import colors
import numpy as np
import lab4

files = [
    '../DAQ/0508_cs_rimbalzi_nopb2.txt',
    '../DAQ/0508_cs_rimbalzi_pb.txt'
]

labels = [
    'senza piombo',
    'con piombo'
]

fig = plt.figure('rimb')
fig.clf()
fig.set_tight_layout(True)

axs = fig.subplots(1, len(files), sharey=True, sharex=True)

norm = colors.LogNorm()

for i in range(len(files)):
    ch1, ch2 = lab4.loadtxt(files[i], unpack=True, usecols=(0,1))
    
    ax = axs[i]
    _, _, _, im = ax.hist2d(ch1, ch2, bins=np.arange(0, 1150, 8), cmap='jet', norm=norm)
    
    if i == 0:
        im0 = im
    if i == len(files) - 1:
        fig.colorbar(im0, ax=ax)
    if i == 0:
        ax.set_ylabel('canale ADC PMT 2')
    ax.set_xlabel('canale ADC PMT 1')
    ax.legend(title=labels[i], loc='upper right')

fig.show()
