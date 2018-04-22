import mc
import lab
import numpy as np

def rutherford(theta):
    return 1 / (1 - np.cos(np.radians(theta))) ** 2

def y(x, xmin):
    return x + np.sign(x) * xmin * np.exp(-np.abs(x) / xmin)

def gauss(x, sigma):
    return np.exp(-(x / sigma) ** 2)

def function(x, xmin, sigma, ampl):
    return rutherford(y(x, xmin)) + ampl * gauss(x, sigma)

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    
    fig = plt.figure('empirical')
    fig.clf()
    fig.set_tight_layout(True)

    ax = fig.add_subplot(111)

    theta = np.linspace(-85, 85, 1000)
    p = (5, 5, 100000)
    ax.semilogy(theta, function(theta, *p), '.')
    ax.plot(theta, rutherford(theta), '-', scaley=False, label='rutherford')
    ax.plot(theta, p[2] * gauss(theta, p[1]), '-', scaley=False, label='gauss')
    
    ax.legend(loc=1)
    ax.grid(linestyle=':')
    
    fig.show()
