#!/usr/bin/env python
from __future__ import division,print_function
from pylab import *
import sys
import lab4
from uncertainties import ufloat as uf

filename=sys.argv[1]

ch1,ch2,ch3,tr1,tr2,tr3,c2,c3,ch2t,ts=lab4.loadtxt(filename,unpack=True,usecols=(0,1,2,4,5,6,8,9,11,12))

out1=ch1[(ch2 > 100)]
out2=ch1[(ch2 < 100)]
out3=ch1

tutti=arange(0,max(ch1))

figure('doppio_picco_log')
clf()

hist(out1,bins=tutti,label="ch1 c2 n=%d"%len(out1),histtype="step")
hist(out2,bins=tutti,label="ch1 notc2 n=%d"%len(out2),histtype="step")
#hist(out3,bins=tutti,label="ch1 n=%d"%len(out3),histtype="step")
legend(loc=0)
yscale('log')

show()

n_c2_tr = len(ch1[(tr1>500)&(tr2>500)])
n_c2_tr=uf(n_c2_tr,sqrt(n_c2_tr))

n_c2 = len(ch1[(c2>500)])
n_c2=uf(n_c2,sqrt(n_c2))

n_c3_tr = len(ch1[(tr1>500)&(tr2>500)&(tr3>500)])
n_c3 = len(ch1[(c3>500)])
coer=(n_c2_tr-n_c2)/n_c2_tr

print("#tr2 =",len(tr2))
print("#c2_tr = {:d}, #c2 = {:d} \ncoer = {:.6f} +- {:.6f}".format(int(n_c2_tr.n),int(n_c2.n),coer.n,coer.s))
print("#c3_tr = %d, #c3 = %d"%(n_c3_tr,n_c3))
