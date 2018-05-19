import numpy as np
from matplotlib import pyplot as plt
import glob
import lab4
import scanf

fig = plt.figure('tdccal')
fig.clf()
fig.set_tight_layout(True)
ax = fig.add_subplot(111)

files = glob.glob('../DAQ/0517_*ns_provatdc.txt')
nominal, tdc = np.empty((2, len(files)))
for i in range(len(files)):
    samples, = lab4.loadtxt(files[i], unpack=True, usecols=(0,), dtype=int)
    nominal[i], = scanf.scanf('../DAQ/0517_%dns_provatdc.txt', s=files[i])
    tdc[i] = np.mean(samples)

ax.plot(nominal, tdc, '.')
ax.set_xlabel('Valore nominale (generatore di forme) [ns]')
ax.set_ylabel('Valore tdc [digit]')
ax.grid(linestyle=':')

fig.show()
