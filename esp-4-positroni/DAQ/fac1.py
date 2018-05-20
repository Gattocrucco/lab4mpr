## Leggere il primo facoltativo
from pylab import *
import lab4
import sys
from matplotlib.colors import LogNorm

file=sys.argv[1]

t1,t2=lab4.loadtxt(file,usecols=(0,1),unpack=True)

delta1=t1-t2

delta1=delta1[(t1!=0) | (t2!=0)]
bins=arange(min(delta1), max(delta1)+1)
m = (410 - 10) / 517
q = 10

#delta1 = (delta1 - q) / m
#bins = (bins - q) / m

figure("tdc").set_tight_layout(True)
clf()

title("Facoltativo 1")
xlabel("t1-t2  [ns]")
ylabel("occorrenze")

hist(delta1,bins=bins,histtype='step',label='delta')
hist(t1[t1!=0],bins=arange(max(t1+1)),histtype='step',label='t1')
hist(t2[t2!=0],bins=arange(max(t2+1)),histtype='step',label='t2')

legend(loc=0)

figure('tdc_scatter').set_tight_layout(True)
clf()
title("Scatter plot coincidenze a 2")
_,_,_,im=plt.hist2d(t1[(t1!=0) | (t2!=0)],t2[(t1!=0) | (t2!=0)],bins=arange(0,520),norm=LogNorm(),cmap='jet')
colorbar(im)
show()



