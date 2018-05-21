#!/usr/bin/env python
from __future__ import division,print_function
from pylab import *
import sys
import lab4
import numpy as np
from matplotlib.colors import LogNorm

filename=sys.argv[1]
if len(sys.argv) >= 4:
  chans = [int(i) for i in sys.argv[2:4]]
else:
  chans = [0, 1]

data = lab4.loadtxt(filename, unpack=True, usecols=chans)

bins = arange(np.max(data) + 1)

figure('histo2d 1d')
clf()

t_slice = 4

#for m in arange(0,t_slice):
#  hist(out[m*len(ch1)//t_slice:(m+1)*len(ch1)//t_slice],bins=tutti,label="ch1 slice=%d"%m,histtype="step")
for i in range(len(chans)):
  hist(data[i], bins=bins, label="ADC a%d" % (chans[i],), histtype="step")

title(filename + ', n=' + str(data.shape[1]))
legend(loc=0)
yscale('log')

figure('histo2d 2d')
clf()
title(filename + ', n=' + str(data.shape[1]))
bins = arange(np.max(data) + 1)[::8]
_,_,_,im=plt.hist2d(data[0], data[1], bins=bins, norm=LogNorm(), cmap='jet')
colorbar(im)
xlabel("ADC a%d" % chans[0])
ylabel("ADC a%d" % chans[1])

show()
