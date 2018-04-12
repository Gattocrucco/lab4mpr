# carica il file di bob per la stopping power

import numpy as np
import lab4

T, dedx = np.loadtxt('../bob/stopping_power_au.txt', unpack=True)
rho = 19.320 # densità dell'oro

f = lab4.interp(T, dedx * rho)

if __name__ == '__main__':
    from matplotlib import pyplot as plt
    fig = plt.figure('dedx')
    fig.clf()
    fig.set_tight_layout(True)

    ax = fig.add_subplot(111)
    ax.plot(T, dedx * rho, '.k')
    x = np.linspace(np.min(T), np.max(T), 10000)
    ax.plot(x, [f(x) for x in x], '-k')

    ax.set_xlabel('energia [MeV]')
    ax.set_ylabel('dE/dx [MeV / cm]')
    ax.set_title('Perdita di energia particelle $\\alpha$')
    ax.grid(linestyle=':')

    fig.show()
