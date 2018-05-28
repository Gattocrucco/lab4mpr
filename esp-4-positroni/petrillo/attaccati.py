import lab4
from pylab import *
from matplotlib import colors

ch1, ch2 = lab4.loadtxt('../DAQ/0522_prova_rtot_attaccati.txt', unpack=True, usecols=(0, 11))

figure('attaccati').set_tight_layout(True)
clf()

_, _, _, im = hist2d(ch1, ch2, bins=arange(0, 1150, 8), cmap='jet', norm=colors.LogNorm())
colorbar(im)

xlabel('energia PMT 1 [canale ADC]')
ylabel('energia PMT 2 [canale ADC]')

show()
