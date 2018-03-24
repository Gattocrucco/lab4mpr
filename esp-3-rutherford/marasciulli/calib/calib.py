## RISOLUZIONE
from pylab import *
import lab4
import lab

t,v=loadtxt("../de0_data/0316ang0.dat",unpack=True,usecols=(0,1))

bordi=arange(0,2**12+1)
# binnaggio preso da bar_line, histogram e compagnia bella
cont,edg=histogram(v,bins=bordi)
errorbar(edg[:-1],cont,sqrt(cont),linestyle="",marker=".",capsize=2)

def gauss(x,N,u,s):
    return N/(s*sqrt(2*pi))*exp(-((x-u)**2)/(2*s**2) )
    
val=[10**4,3100,100]
bordi=bordi[:-1]
out=lab.fit_curve(gauss,bordi[cont>3],cont[cont>3],dy=sqrt(cont[cont>3]),p0=val,absolute_sigma=True,print_info=1)

z=linspace(2900,3300,1000)
plot(z,gauss(z,*out.par),color="red")

show()