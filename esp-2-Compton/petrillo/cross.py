import numpy as np
import numba as nb
import matplotlib.pyplot as plt

data = np.loadtxt('../bob/photon_cross_section_NaI.txt', unpack=True)

def interp(x, y):
    """
    x must be sorted
    """
    @nb.jit('f8(f8)', nopython=True)
    def fun(x0):
        assert x[0] <= x0 <= x[-1]
        idx = np.searchsorted(x, x0)
        # we have x[idx-1] < x0 <= x[idx]
        if idx == 0:
            idx = 1
        center_x = x[idx - 1]
        center_y = y[idx - 1]
        slope = (y[idx] - center_y) / (x[idx] - center_x)
        return slope * (x0 - center_x) + center_y
    
    return fun

def explog(f):
    @nb.jit('f8(f8)', nopython=True)
    def fun(x):
        return np.exp(f(np.log(x)))
    return fun

rayleigh = explog(interp(np.log(data[0]), np.log(data[1])))
compton  = explog(interp(np.log(data[0]), np.log(data[2])))
photoel  = explog(interp(np.log(data[0]), np.log(data[3])))
couple   = explog(interp(np.log(data[0]), np.log(data[4])))
total    = explog(interp(np.log(data[0]), np.log(np.sum(data[1:], axis=0))))

if __name__ == '__main__':
    fig = plt.figure('cross')
    fig.clf()
    fig.set_tight_layout(True)
    ax = fig.add_subplot(111)

    e = np.logspace(np.log10(np.min(data[0])), np.log10(np.max(data[0])), 1000)
    ax.plot(e, [total   (E) for E in e],  '-',     color='black', label='Totale')
    ax.plot(e, [photoel (E) for E in e], '--',     color='black', label='Fotoelettrico')
    ax.plot(e, [compton (E) for E in e],  ':',     color='black', label='Compton')
    ax.plot(e, [rayleigh(E) for E in e], '--', color='lightgray', label='Rayleigh')
    ax.plot(e, [couple  (E) for E in e],  ':', color='lightgray', label='Coppie')
    ax.legend()
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Energia del fotone [MeV]')
    ax.set_ylabel('Coefficiente di attenuazione di massa [cm$^{2}$ g$^{-1}$]')
    ax.set_xlim((np.min(e), np.max(e)))
    y = ax.get_ylim()
    ax.set_ylim((np.nanmin([couple  (E) for E in e]), y[1]))
    ax.grid()

    fig.show()
