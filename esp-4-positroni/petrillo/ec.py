import numpy as np
from matplotlib import pyplot as plt
from matplotlib import gridspec
from matplotlib import colors
import lab4

file_c2 = '../DAQ/0511_ec2_c2.txt'
file_ch1 = '../DAQ/0511_ec2_ch1.txt'
file_ch2 = '../DAQ/0511_ec2_ch2.txt'

# prepare figures
fig_c2 = plt.figure('ec-c2')
fig_c2.clf()
fig_c2.set_tight_layout(True)
G = gridspec.GridSpec(3, 3)
ax_c2 = fig_c2.add_subplot(G[:-1,1:])
ax_c2_ch1 = fig_c2.add_subplot(G[-1,1:], sharex=ax_c2)
ax_c2_ch2 = fig_c2.add_subplot(G[:-1,0], sharey=ax_c2)
ax_colorbar = fig_c2.add_subplot(G[-1,0])

fig_ch = plt.figure('ec-ch')
fig_ch.clf()
fig_ch.set_tight_layout(True)
ax_ch1 = fig_ch.add_subplot(121)
ax_ch2 = fig_ch.add_subplot(122)

# load data
ch1_c2, ch2_c2 = lab4.loadtxt(file_c2, unpack=True, usecols=(0, 1))
ch1, = lab4.loadtxt(file_ch1, unpack=True, usecols=(0,))
ch2, = lab4.loadtxt(file_ch2, unpack=True, usecols=(1,))

# plot
bins = np.arange(1150 // 8) * 8
H, _, _, im = ax_c2.hist2d(ch1_c2, ch2_c2, bins=bins, norm=colors.LogNorm(), cmap='jet')
fig_c2.colorbar(im, ax=ax_colorbar, fraction=0.5, aspect=2)
ax_colorbar.axis('off')
ax_c2_ch1.hist(ch1_c2, bins=bins, orientation='vertical', histtype='step', log=True)
ax_c2_ch2.hist(ch2_c2, bins=bins, orientation='horizontal', histtype='step', log=True)
ax_ch1.hist(ch1, bins=bins, histtype='step', log=True)
ax_ch2.hist(ch2, bins=bins, histtype='step', log=True)

ax_c2_ch1.set_xlabel('energia PMT 1 [canale ADC]')
ax_c2_ch2.set_ylabel('energia PMT 2 [canale ADC]')

ax_ch1.set_xlabel('energia PMT 1 [canale ADC]')
ax_ch2.set_xlabel('energia PMT 2 [canale ADC]')

# fit

fig_ch.show()
fig_c2.show()
