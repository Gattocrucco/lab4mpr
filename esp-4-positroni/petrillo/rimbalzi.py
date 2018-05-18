import numpy as np
from matplotlib import pyplot as plt

E = 662

def compton_energy(E, cos_theta):
    """
    E in keV
    """
    return E / (1 + E/511 * (1-cos_theta))

fig = plt.figure('rimbalzi')
fig.clf()
ax = fig.add_subplot(111)

thetaspace = np.linspace(0, 90)
ax.plot(thetaspace, compton_energy(E, np.cos(np.radians(thetaspace))), label='Fotone rimbalzato che arriva')
ax.plot(thetaspace, E - compton_energy(E, np.cos(np.radians(thetaspace))), label='Fotone che rimbalza')

ax.set_xlabel('angolo [°]')
ax.set_ylabel('energia rilasciata')
ax.legend(loc='best')
ax.grid(linestyle=':', which='both')

fig.show()
