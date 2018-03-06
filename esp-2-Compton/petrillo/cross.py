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

log_compton = interp(np.log(data[0]), np.log(data[2]))
log_photoel = interp(np.log(data[0]), np.log(data[3]))

compton = explog(log_compton)
photoel = explog(log_photoel)

if __name__ == '__main__':
    fig = plt.figure('cross')
    fig.clf()
    ax = fig.add_subplot(111)

    e = np.logspace(np.log10(np.min(data[0])), np.log10(np.max(data[0])), 500)
    ax.plot(data[0], data[2], '.', label='compton')
    ax.plot(data[0], data[3], '.', label='photoel')
    ax.plot(e, [compton(E) for E in e], label='compton')
    ax.plot(e, [photoel(E) for E in e], label='photoel')
    ax.legend()
    ax.set_xscale('log')
    ax.set_yscale('log')

    fig.show()
