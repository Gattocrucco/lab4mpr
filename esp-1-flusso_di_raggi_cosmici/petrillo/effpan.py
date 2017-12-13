import numpy as np
import matplotlib.pyplot as plt
from montecarlo import MC, pmt
import lab
from uncertainties import ufloat

mc = MC(pmt(6), pmt(5), pmt(2), pmt(1))
mc.random_ray(1e4)

N = 1000

ce6 = np.empty(N)
ce1 = np.copy(ce6)

eta = lab.Eta()
for i in range(N):
    eta.etaprint(i / N)
    
    mc.run(pivot_scint=1, randgeom=True)
    
    c51 = mc.count(None, 1, None, 1)
    c6521 = mc.count(1, 1, 1, 1)
    c652 = mc.count(1, 1, 1, None)
    
    ce6[i] = (c51 / c6521).n
    ce1[i] = (c652 / c6521).n

f6 = ufloat(np.mean(ce6), np.std(ce6) / np.sqrt(len(ce6)))
f1 = ufloat(np.mean(ce1), np.std(ce1) / np.sqrt(len(ce1)))

print(f6)
print(f1)

plt.figure('effpan')
plt.clf()

plt.subplot(211)
plt.title('c51 / c651')
plt.hist(ce6, bins='sqrt')

plt.subplot(212)
plt.title('c62 / c621')
plt.hist(ce1, bins='sqrt')

plt.show()
