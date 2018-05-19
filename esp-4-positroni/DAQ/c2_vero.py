#!/usr/bin/env python
from __future__ import division,print_function
from pylab import *
import sys
import lab4
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

filename=sys.argv[1]

ch1,ch2,ch3,tr1,tr2,tr3,c2,c3,ts=lab4.loadtxt(filename,unpack=True,usecols=(0,1,2,4,5,6,8,9,12))

out1=ch1[(ch1>100)]
out2=ch2[(ch2>100)]
out3=ch3[(tr3>500)]

out1_2=ch1[(ch1>100) & (ch2>100)]
out2_2=ch2[(ch2>100) & (ch1>100)]
out3_2=ch3[(tr3>500) & (c2>500)]
'''
out1_3=ch1[(tr1>500) & (c3>500)]
out2_3=ch2[(tr2>500) & (c3>500)]
out3_3=ch3[(tr3>500) & (c3>500)]
'''
tutti=arange(0,max(ch1))


# istogrammi e scatter 2

figure('c2').set_tight_layout(True)
clf()
'''
subplot(121)
title("Acquisizioni singole")
hist(out1,bins=tutti,label="ch1 n=%d"%len(out1),histtype="step")
hist(out2,bins=tutti,label="ch2 n=%d"%len(out2),histtype="step")
#hist(out3,bins=tutti,label="ch3 n=%d"%len(out3),histtype="step")
legend(loc=0)
yscale('log')
'''
subplot(111)
title("Coincidenze a 2")
hist(out1_2,bins=tutti,label="ch1 c2 n=%d"%len(out1_2),histtype="step")
hist(out2_2,bins=tutti,label="ch2 c2 n=%d"%len(out2_2),histtype="step")
#if len(out3_3)>0:
#    hist(out3_2,bins=tutti,label="ch3 c2 n=%d"%len(out3_2),histtype="step")
legend(loc=0,fontsize='small')
yscale('log')


figure('sc2').set_tight_layout(True)
clf()
title("Scatter plot coincidenze a 2")
_,_,_,im=plt.hist2d(out1_2,out2_2,bins=tutti,norm=LogNorm(),cmap='jet')
colorbar(im)
xlabel("ch1")
ylabel("ch2")
xlim(min(out1_2)-5,max(out1_2)+5)
ylim(min(out2_2)-5,max(out2_2)+10)

show()
