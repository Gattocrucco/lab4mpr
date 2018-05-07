#!/usr/bin/env python
from __future__ import division,print_function
from pylab import *
import sys
import lab4

filename=sys.argv[1]

ch1,ch2,ch3,tr1,tr2,tr3,c2,c3,ts=lab4.loadtxt(filename,unpack=True,usecols=(0,1,2,4,5,6,8,9,12))

figure('temp ch1')

plot((ts-ts[0])/3600, ch1,marker='.',markersize=2,linestyle='',label=filename)

figure("temp ch2")
plot((ts-ts[0])/3600, ch2,marker='.',markersize=2,linestyle='',label=filename)

legend()
show()
