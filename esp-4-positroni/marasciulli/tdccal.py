import numpy as np
from matplotlib import pyplot as plt
import glob
import lab4
import scanf
import lab

fig = plt.figure('tdccal')
fig.clf()
fig.set_tight_layout(True)
ax = fig.add_subplot(111)

ax.set_title('Calibrazione TDC scala 102 ns')
 
# caricamento dei file
files = glob.glob('../DAQ/0517_*ns_provatdc.txt')
nominal, tdc, stdc = np.empty((3, len(files)))
for i in range(len(files)):
    files[i] = files[i].replace("\\", '/')
    samples, = lab4.loadtxt(files[i], unpack=True, usecols=(0,), dtype=int)
    nominal[i], = scanf.scanf('../DAQ/0517_%dns_provatdc.txt', s=files[i])
    par,cov=lab.fit_oversampling(samples)
    tdc[i]=par[0]
    stdc[i]=np.sqrt(cov[0,0])

ax.errorbar(nominal,tdc,stdc,fmt='.k',capsize=2)
ax.set_xlabel('Valore nominale (generatore di forme) [ns]',size=12)
ax.set_ylabel('Valore tdc [digit]',size=12)
ax.grid(linestyle=':')

# fit dei dati lineari

def retta(x,m,q):
    return m*x+q

fnominal=np.sort(nominal)
ftdc=tdc[np.argsort(nominal)]
fstdc=stdc[np.argsort(nominal)]

stime=(1,0)
fit=lab.fit_curve(retta,fnominal[1:-1],ftdc[1:-1],dy=fstdc[1:-1],p0=stime,print_info=True)

space=np.linspace(0,130)
ax.plot(space,retta(space,*fit.par),color='red')

fig.show()
