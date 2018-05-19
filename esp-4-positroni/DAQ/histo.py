#!/usr/bin/env python
from __future__ import division,print_function
from pylab import *
import sys
import lab4

filename=sys.argv[1]

ch1,ch2,ch3,tr1,tr2,tr3,c2,c3,ts=lab4.loadtxt(filename,unpack=True,usecols=(0,1,2,4,5,6,8,9,12))

out1=ch1[ch1>100]
out2=ch2[ch2>100]
#out3=ch3[(tr3>500)]

tutti=arange(0,max(ch1)+1)
#tutti=arange(0,1200//8)*8

figure('histo')
clf()

#t_slice = 1

#for m in arange(0,t_slice):
#  hist(out1[m*len(ch1)//t_slice:(m+1)*len(ch1)//t_slice],bins=tutti,label="ch1 slice=%d"%m,histtype="step")
hist(out1,bins=tutti,label="ch1 n=%d"%len(out1),histtype="step")
#hist(out2,bins=tutti,label="ch2 n=%d"%len(out2),histtype="step")
legend(loc=0)
yscale('log')

show()
