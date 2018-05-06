#!/usr/bin/env python
from __future__ import division,print_function
from pylab import *
import sys

filename=sys.argv[1]
if sys.platform=='win32':
    from lab4 import loadtxt
    ch1,ch2,ch3,tr1,tr2,tr3,c2,c3,ts=loadtxt(filename,unpack=True,usecols=(0,1,2,4,5,6,8,9,12))
else:
    ch1,ch2,ch3,tr1,tr2,tr3,c2,c3,ts=loadtxt(filename,unpack=True,usecols=(0,1,2,4,5,6,8,9,12))

out1=ch1[(tr1>500)]
out2=ch2[(tr2>500)]
out3=ch3[(tr3>500)]

out1_2=ch1[(tr1>500) & (c2>500)]
out2_2=ch2[(tr2>500) & (c2>500)]
out3_2=ch3[(tr3>500) & (c2>500)]

out1_3=ch1[(tr1>500) & (c3>500)]
out2_3=ch2[(tr2>500) & (c3>500)]
out3_3=ch3[(tr3>500) & (c3>500)]

#tuttivecchio=arange(0,max(tr1))
tutti=arange(0,1200//8)*8

# istogrammi

figure(1).set_tight_layout(True)

subplot(121)
title("Acquisizioni singole")
hist(out1,bins=tutti,label="ch1 n=%d"%len(out1),histtype="step")
hist(out2,bins=tutti,label="ch2 n=%d"%len(out2),histtype="step")
hist(out3,bins=tutti,label="ch3 n=%d"%len(out3),histtype="step")
legend(loc=0)


subplot(122)
title("Coincidenze a 3")
hist(out1_3,bins=tutti,label="ch1 c3 n=%d"%len(out1_3),histtype="step")
hist(out2_3,bins=tutti,label="ch2 c3 n=%d"%len(out2_3),histtype="step")
hist(out3_3,bins=tutti,label="ch3 c3 n=%d"%len(out3_3),histtype="step")
legend(loc=0)

figure(2).set_tight_layout(True)

subplot(221)
title("Coincidenze a 3: CH1 vs CH2")
plot(out1_3,out2_3,linestyle='',marker='.',markersize=1)
xlabel("ch1")
ylabel('ch2')

subplot(222)
title("Coincidenze a 3: CH2 vs CH3")
plot(out2_3,out3_3,linestyle='',marker='.',markersize=1)
xlabel("ch2")
ylabel('ch3')

subplot(223)
title("Coincidenze a 3: CH1 vs CH3")
plot(out1_3,out3_3,linestyle='',marker='.',markersize=1)
xlabel("ch1")
ylabel('ch3')

show()