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

tre1=ch1[(tr1>500) & (c3>500)]
tre2=ch2[(tr2>500) & (c3>500)]
out6=ch3[(tr3>500) & (c3>500)]

tuttivecchio=arange(0,max(tr1))
tutti=arange(0,1200//8)*8

figure(1)

hist(out1,bins=tutti,label="ch1 n=%d"%len(out1),histtype="step")
hist(out2,bins=tutti,label="ch2 n=%d"%len(out2),histtype="step")
hist(out3,bins=tutti,label="ch3 n=%d"%len(out3),histtype="step")
legend(loc=0)

figure(2)

due3=ch3[(tr3>500) & (c2>500)]

hist(out4,bins=tutti,label="ch1 c2 n=%d"%len(out4),histtype="step")
hist(out5,bins=tutti,label="ch2 c2 n=%d"%len(out5),histtype="step")
#hist(due3,bins=tutti,label="ch3 c2 n=%d"%len(due3),histtype="step")
legend(loc=0)

figure(3)

hist(tre1,bins=tutti,label="ch1 c3 n=%d"%len(tre1),histtype="step")
hist(tre2,bins=tutti,label="ch2 c3 n=%d"%len(tre2),histtype="step")
hist(out6,bins=tutti,label="ch3 c3 n=%d"%len(out6),histtype="step")

legend(loc=0)


figure(4)

sc1=ch1[c2>500]
sc2=ch2[c2>500]

title("scatter plot coincidenze a 2")
plot(sc1,sc2,linestyle='',marker='.',markersize=1)
xlabel("ch1")
ylabel("ch2")

figure(5)

sc33=ch3[c3>500]
sc13=ch1[c3>500]
sc23=ch2[c3>500]

plot(sc13,sc23,linestyle='',marker='.',markersize=1)
xlabel("ch1")
ylabel('ch2')

figure(6)

plot(sc23,sc33,linestyle='',marker='.',markersize=1)
xlabel("ch2")
ylabel('ch3')

figure(7)

plot(sc13,sc33,linestyle='',marker='.',markersize=1)
xlabel("ch1")
ylabel('ch3')

show()
