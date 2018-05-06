import numpy as np
import lab
import lab4
from matplotlib import pyplot as plt

# arguments
filename = '../DAQ/0504_cal_na_ch1.txt'
source = 'na'
channel = 1

# load data
ch1, ch2, ch3, tr1, tr2, tr3, c2, c3, ts = lab4.loadtxt(filename, unpack=True, usecols=(0, 1, 2, 4, 5, 6, 8, 9, 12))
samples = np.array([ch1, ch2, ch3][channel - 1][[tr1, tr2, tr3][channel - 1] > 500], dtype=int)
hist = np.bincount(samples)
bins = np.arange(len(hist) + 1)
missing_codes = bins[(bins // 4) % 2 == 0]

# fit


# plot
fig = plt.figure('calibration')
fig.clf()
ax = fig.add_subplot(111)

lab4.bar(bins, hist, ax=ax)

fig.show()
