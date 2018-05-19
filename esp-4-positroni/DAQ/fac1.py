## Leggere il primo facoltativo
from pylab import *
import lab4
import sys

file=sys.argv[1]

t1,t2=lab4.loadtxt(file,usecols=(0,1),unpack=True)

delta1=t1-t2
delta1=delta1[(t1!=0) | (t2!=0)]
bins=arange(min(delta1), max(delta1)+1)

m = (450 - 30) / 500
q = 30

delta1 = (delta1 - q) / m
bins = (bins - q) / m

figure("tdc").set_tight_layout(True)
clf()

title("Facoltativo 1")
xlabel("t1-t2  [ns]")
ylabel("occorrenze")

hist(delta1,bins=bins,histtype='step')

show()



