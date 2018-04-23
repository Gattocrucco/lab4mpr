# carica il file di bob per la stopping power

import numpy as np
import lab4

T, dedx_data = np.loadtxt('../bob/stopping_power_au.txt', unpack=True)

rho_au = 19.320 # densità dell'oro
rho_al = 2.699

dedx = lab4.interp(T, dedx_data)
dedx_min = np.min(T)

if __name__ == '__main__':
    from matplotlib import pyplot as plt
    fig = plt.figure('dedx')
    fig.clf()
    fig.set_tight_layout(True)

    ax = fig.add_subplot(111)
    ax.plot(T, dedx_data * rho_au / 10000, '.k')
    x = np.linspace(np.min(T), np.max(T), 10000)
    ax.plot(x, [dedx(x) * rho_au / 10000 for x in x], '-k')

    ax.set_xlabel('energia [MeV]')
    ax.set_ylabel('dE/dx [MeV $\\mu$m$^{-1}$]')
    ax.set_title('Perdita di energia particelle $\\alpha$ nell\'oro')
    ax.grid(linestyle=':')

    fig.show()
