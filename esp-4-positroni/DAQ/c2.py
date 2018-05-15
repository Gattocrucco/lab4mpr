#!/usr/bin/env python
from __future__ import division,print_function
from pylab import *
import sys
import lab4
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

# read command line
filename=sys.argv[1]
if len(sys.argv) >= 3:
  coinc_str = sys.argv[2]
else:
  coinc_str = '12'
coinc = {'12': [0,1], '13': [0,2], '23': [1,2]}[coinc_str]

# load data
ch1,ch2,ch3,tr1,tr2,tr3,c2,c3,ts=lab4.loadtxt(filename,unpack=True,usecols=(0,1,2,4,5,6,8,9,12))

ch = array([ch1, ch2, ch3])
tr = array([tr1, tr2, tr3])

cha, chb = ch[coinc]
tra, trb = tr[coinc] > 500
outa = cha[tra]
outb = chb[trb]

c2 = c2 > 500
c2_tr = tra & trb

outa_2=cha[c2_tr]
outb_2=chb[c2_tr]

tutti=arange(0,1200//8)*8

# istogrammi e scatter 2

figure('c2').set_tight_layout(True)
clf()
'''
subplot(121)
title("Acquisizioni singole")
hist(out1,bins=tutti,label="ch1 n=%d"%len(out1),histtype="step")
hist(out2,bins=tutti,label="ch2 n=%d"%len(out2),histtype="step")
hist(out3,bins=tutti,label="ch3 n=%d"%len(out3),histtype="step")
legend(loc=0)
'''

title("Coincidenze a 2")
hist(outa_2,bins=tutti,label="ch%d c2 n=%d"%(coinc[0]+1, len(outa_2)),histtype="step")
hist(outb_2,bins=tutti,label="ch%d c2 n=%d"%(coinc[1]+1, len(outb_2)),histtype="step")
#if len(out3_3)>0:
#    hist(out3_2,bins=tutti,label="ch3 c2 n=%d"%len(out3_2),histtype="step")
legend(loc=0,fontsize='small')
yscale('log')


figure('sc2').set_tight_layout(True)
clf()
title("Scatter plot coincidenze a 2")
_,_,_,im=plt.hist2d(outa_2,outb_2,bins=tutti,norm=LogNorm(),cmap='jet')
colorbar(im)
xlabel("ch%d" % (coinc[0] + 1))
ylabel("ch%d" % (coinc[1] + 1))
xlim(min(outa_2)-5,max(outa_2)+5)
ylim(min(outb_2)-5,max(outb_2)+10)

show()
n_c2_tr = np.sum(c2_tr)
n_c2 = np.sum(c2)

print("#c2_tr = %d, #c2 = %d"%(n_c2_tr,n_c2))

