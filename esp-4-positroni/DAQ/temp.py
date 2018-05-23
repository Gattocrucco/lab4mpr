#!/usr/bin/env python
from __future__ import division, print_function
from pylab import *
import sys
import lab4
import numpy as np

# usage:
# temp.py <file>
# temp.py <file> <canali> ...

filename=sys.argv[1]
if len(sys.argv) >= 3:
  chans = [int(i) for i in sys.argv[2:]]
else:
  chans = [0]

data = lab4.loadtxt(filename, unpack=True, usecols=chans + [12])
ts = data[-1]
ts -= ts[0]

decimation = data.shape[1] // 10000
ts = ts[::decimation]

figure('temp')
clf()

for i in range(len(chans)):
  subplot(len(chans), 1, i + 1)
  if i == 0:
    title(filename + ', n=' + str(data.shape[1]) + (' [::%d]' % decimation if decimation else ''))
  plot(ts, data[i,::decimation], ',', label="ADC a%d" % (chans[i],))
  legend(loc=0)
xlabel('tempo [s]')
show()

