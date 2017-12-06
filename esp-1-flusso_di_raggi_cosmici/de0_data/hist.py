#!/usr/bin/env python

from __future__ import division, print_function
from pylab import *
import sys

files = sys.argv[1:]
maxbins = int(sqrt(10000))

figure('hist')
clf()

ch0s = []

for file in files:
	t, ch0, ch1 = loadtxt(file, unpack=True)
	ch0s.append(ch0)

maxlen = max([len(ch0) for ch0 in ch0s])
hist(ch0s, normed=True, bins=min(maxbins, int(sqrt(maxlen))), label=files, alpha=.5)

legend(loc=0, fontsize='small')
show()
