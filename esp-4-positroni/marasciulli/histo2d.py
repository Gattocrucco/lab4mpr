import numpy as np
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
import sys
import lab4

try:
    filename = sys.argv[1]
except IndexError:
    raise IndexError("Devi lanciarlo da linea di comando e scrivere il nome del file.")
ch1, ch2, c2 = lab4.loadtxt(filename, usecols=(0, 1, 8), unpack=True)

fig = plt.figure('histo2d')
fig.clf()
fig.tight_layout(True)

ax = fig.add_subplot(111)

H,_,_,im=plt.hist2d(ch1[c2 > 500], ch2[c2 > 500],bins=np.arange(0,1200//8)*8,norm=LogNorm(),cmap='jet')
ax.set_xlabel('ch1 c2')
ax.set_ylabel('ch2 c2')
fig.colorbar(im)

fig.show()
