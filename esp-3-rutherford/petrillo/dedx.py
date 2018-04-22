# carica il file di bob per la stopping power

import numpy as np
import lab4

T_au, dedx_au_data = np.loadtxt('../bob/stopping_power_au.txt', unpack=True)
T_al, _, _, dedx_al_data = np.loadtxt('../bob/stopping_power_al.txt', unpack=True)

rho_au = 19.320 # g/cm^3
rho_al = 2.699 # g/cm^3

dedx_au = lab4.interp(T_au, dedx_au_data)
dedx_al = lab4.interp(T_al, dedx_al_data)
dedx_min = max(np.min(T_au), np.min(T_al))

dedx = {
    13: dedx_al,
    79: dedx_au
}

if __name__ == '__main__':
    from matplotlib import pyplot as plt
    fig = plt.figure('dedx')
    fig.clf()
    fig.set_tight_layout(True)

    ax = fig.add_subplot(111)
    ax.plot(T_au, dedx_au_data, '.', label='Au data')
    ax.plot(T_al, dedx_al_data, '.', label='Al data')
    x = np.linspace(np.min(T_au), np.max(T_au), 10000)
    ax.plot(x, [dedx_au(x) for x in x], '-k')
    ax.plot(x, [dedx_al(x) for x in x], '-k')

    ax.set_xlabel('energia [MeV]')
    ax.set_ylabel('dE/dx [MeV g$^{-1}$ cm$^{2}$]')
    ax.set_title('Perdita di energia particelle $\\alpha$')
    ax.grid(linestyle=':')
    ax.set_xlim(0, 10)
    ax.legend(loc=1)

    fig.show()
