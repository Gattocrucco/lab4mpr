#!/usr/bin/env python
from __future__ import division,print_function
from pylab import *
import sys
import lab4
#from scipy.optimize import curve_fit

figure('histo2-ch1')
clf()
figure('histo2-ch2')
clf()
figure('histo2-ch3')
clf()

for i in range(1,len(sys.argv)):

  ch1,ch2,ch3,tr1,tr2,tr3,c2,c3,ts=lab4.loadtxt(sys.argv[i],unpack=True,usecols=(0,1,2,4,5,6,8,9,12))
  #a2ch1,a2ch2,a2ch3,a2tr1,a2tr2,a2tr3,a2c2,a2c3,ts=loadtxt(sys.argv[2],unpack=True,usecols=(0,1,2,4,5,6,8,9,12))

  #out1=ch1[(tr1>500)]
  out1=ch1
  out2=ch2[(tr2>500)]
  out3=ch3[(tr3>500)]
  #a2out1=a2ch1[(a2tr1>500) & (a2c2>500)]
  #a2out2=a2ch2[(a2tr2>500) & (a2c2>500)]


  tutti=arange(0,1200)
  #tutti=arange(0,1200//8)*8

  figure('histo2-ch1')
  
  title("Istogrammi normalizzati")
  
  hist(out1,bins=tutti,label="ch1 %s"%(sys.argv[i]),histtype="step",density=True)
  #hist(a2out1,bins=tutti,label="ch1 %s %d"%(sys.argv[2],len(a2out1)),histtype="step")
  legend(loc=0,fontsize='small')

  figure('histo2-ch2')

  title("Istogrammi normalizzati")

  hist(out2,bins=tutti,label="ch2 %s"%(sys.argv[i]),histtype="step",density=True)
  #hist(a2out2,bins=tutti,label="ch2 %s %d"%(sys.argv[2],len(a2out2)),histtype="step")
  legend(loc=0,fontsize='small')

  figure('histo2-ch3')

  title("Istogrammi normalizzati")

  hist(out3,bins=tutti,label="ch3 %s"%(sys.argv[i]),histtype="step",density=True)
  #hist(a2out2,bins=tutti,label="ch2 %s %d"%(sys.argv[2],len(a2out2)),histtype="step")
  legend(loc=0,fontsize='small')

show()
