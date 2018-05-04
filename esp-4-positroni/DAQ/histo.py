#!/usr/bin/env python
from __future__ import division,print_function
from pylab import *
import sys

filename=sys.argv[1]

ch1,ch2,ch3,tr1,tr2,tr3,c2,c3,ts=loadtxt(filename,unpack=True,usecols=(0,1,2,4,5,6,8,9,12))

out1=ch1[(tr1>500)]
out2=ch2[(tr2>500)]
out3=ch3[(tr3>500)]
out4=ch1[(tr1>500) & (c2>500)]
out5=ch2[(tr2>500) & (c2>500)]
out6=ch3[(tr3>500) & (c2>500)]

tuttivecchio=arange(0,max(tr1))
tutti=arange(0,1200//8)*8

figure(1)

#hist(out1,bins=tutti,label="ch1 %d"%len(out1),histtype="step")
#hist(out2,bins=tutti,label="ch2 %d"%len(out2),histtype="step")
#hist(out3,bins=tutti,label="ch3 %d"%len(out3),histtype="step")
hist(out4,bins=tutti,label="ch1 c2 %d"%len(out4),histtype="step")
hist(out5,bins=tutti,label="ch2 c2 %d"%len(out5),histtype="step")
#hist(out6,bins=tutti,label="ch3 c2 %d"%len(out6),histtype="step")
legend(loc=0)

figure(2)

sc1=ch1[c2>500]
sc2=ch2[c2>500]

plot(sc1,sc2,linestyle='',marker='.',markersize=2)
xlabel("ch1")
ylabel("ch2")

#legend(loc=0)
show()
