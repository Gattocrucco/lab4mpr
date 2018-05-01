#!/usr/bin/env python
from __future__ import division,print_function
from pylab import *
import sys

filename=sys.argv[1]

ch1,ch2,ch3,tr1,tr2,tr3,c2,c3,ts=loadtxt(filename,unpack=True,usecols=(0,1,2,4,5,6,8,9,12))

out1=ch2[tr2>500]

hist(out1,bins=int(sqrt(len(out1))),label="ch2")

legend(loc=0)
show()
