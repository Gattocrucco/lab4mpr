#!/usr/bin/env python
from __future__ import division,print_function
from pylab import *
import sys
import lab4
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from mpl_toolkits.mplot3d import Axes3D

filename=sys.argv[1]
ch1,ch2,ch3,tr1,tr2,tr3,c2,c3,ts=lab4.loadtxt(filename,unpack=True,usecols=(0,1,2,4,5,6,8,9,12))

out1=ch1[(tr1>500)]
out2=ch2[(tr2>500)]
out3=ch3[(tr3>500)]

out1_2=ch1[(tr1>500) & (c2>500)]
out2_2=ch2[(tr2>500) & (c2>500)]
out3_2=ch3[(tr3>500) & (c2>500)]

#c3 = (tr1 > 500) & (tr2 > 500) & (tr3 > 500)
c3 = c3 > 500

out1_3=ch1[c3]
out2_3=ch2[c3]
out3_3=ch3[c3]

#tuttivecchio=arange(0,max(tr1))
tutti=arange(0,1200//8)*8

# scatter 2

figure('pannello principale').set_tight_layout(True)
clf()
subplot(221)
title("Acquisizioni singole")
hist(out1,bins=tutti,label="ch1 n=%d"%len(out1),histtype="step")
hist(out2,bins=tutti,label="ch2 n=%d"%len(out2),histtype="step")
hist(out3,bins=tutti,label="ch3 n=%d"%len(out3),histtype="step")
legend(loc=0)

subplot(222)
title("Coincidenze a 2")
hist(out1_2,bins=tutti,label="ch1 c2 n=%d"%len(out1_2),histtype="step")
hist(out2_2,bins=tutti,label="ch2 c2 n=%d"%len(out2_2),histtype="step")
if len(out3_3)>0:
    hist(out3_2,bins=tutti,label="ch3 c2 n=%d"%len(out3_2),histtype="step")
legend(loc=0)

subplot(223)
title("Coincidenze a 3")
hist(out1_3,bins=tutti,label="ch1 c3 n=%d"%len(out1_3),histtype="step")
hist(out2_3,bins=tutti,label="ch2 c3 n=%d"%len(out2_3),histtype="step")
if len(out3_3)>0:
    hist(out3_3,bins=tutti,label="ch3 c3 n=%d"%len(out3_3),histtype="step")
legend(loc=0)


subplot(224)
title("Scatter plot coincidenze a 2")
_,_,_,im=plt.hist2d(out1_2,out2_2,bins=tutti,norm=LogNorm(),cmap='jet')
colorbar(im)
xlabel("ch1")
ylabel("ch2")
xlim(min(out1_2)-5,max(out1_2)+5)
ylim(min(out2_2)-5,max(out2_2)+10)


figure('scatterino').set_tight_layout(True)
clf()
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

'''
filg=plt.figure('scatt 3d')
clf()
filg.set_tight_layout(True)
foglio=filg.add_subplot(111,projection='3d')
foglio.scatter(out1_3,out2_3,out3_3, c='blue', marker='.') 
'''
show()
