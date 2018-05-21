#!/usr/bin/env python
from __future__ import division,print_function
from pylab import *
import sys
import lab4
import numpy as np

# usage:
# histo.py <file>
# histo.py <file> <canali> ...

filename=sys.argv[1]
if len(sys.argv) >= 3:
  chans = [int(i) for i in sys.argv[2:]]
else:
  chans = [0]

#ch1,ch2,ch3,tr1,tr2,tr3,c2,c3,ts=lab4.loadtxt(filename,unpack=True,usecols=(0,1,2,4,5,6,8,9,12))
data = lab4.loadtxt(filename, unpack=True, usecols=chans)

bins = arange(np.max(data) + 1)

figure('histo')
clf()

for i in range(len(chans)):
  hist(data[i], bins=bins, label="ADC a%d" % (chans[i],), histtype="step")

title(filename + ', n=' + str(data.shape[1]))
legend(loc=0)
yscale('log')

show()
