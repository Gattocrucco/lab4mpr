#!/usr/bin/env python
from __future__ import division,print_function
from pylab import *
import sys
import lab4
import numpy as np

# usage:
# slice.py <file>                         # -> slices=2, chan=0
# slice.py <numero di slice>              # -> chan=0
# slice.py <numero di slice> <canale>

filename = sys.argv[1]
if len(sys.argv) == 2:
  slices = 2
  chans = [0]
elif len(sys.argv) == 3:
  slices = int(sys.argv[2])
  chans = [0]
elif len(sys.argv) >= 4:
  slices = int(sys.argv[2])
  chans = [int(sys.argv[3])]

#ch1,ch2,ch3,tr1,tr2,tr3,c2,c3,ts=lab4.loadtxt(filename,unpack=True,usecols=(0,1,2,4,5,6,8,9,12))
data = lab4.loadtxt(filename, unpack=True, usecols=chans)
data = data[0]

bins = arange(np.max(data) + 1)

figure('slice')
clf()

#for m in arange(0,t_slice):
#  hist(out[m*len(ch1)//t_slice:(m+1)*len(ch1)//t_slice],bins=tutti,label="ch1 slice=%d"%m,histtype="step")
for i in range(slices):
  hist(data[len(data) // slices * i:len(data) // slices * (i+1)], bins=bins, label="ADC a%d slice %d" % (chans[0], i + 1), histtype="step")

title(filename + ', n=' + str(len(data)))
legend(loc=0)
yscale('log')

show()
