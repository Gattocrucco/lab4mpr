from pylab import *
import numba
import numpy as np

data1 = loadtxt('../de0_data/piombo_con_tutto_c12.dat', unpack=True, usecols=(1,))
data2 = loadtxt('../de0_data/piombo_senza_c12.dat', unpack=True, usecols=(1,))

numba.jit(cache=True, nopython=True)
def oneside(dataa, datab):
    dataa = np.sort(dataa)
    datab = np.sort(datab)
    M = 0.
    j = 0
    for i in range(len(dataa)):
        while j < len(datab) and datab[j] <= dataa[i]:
            j += 1
        cumb = j / len(datab)
        cuma = i / len(dataa)
        D = abs(cumb - cuma)
        if D > M:
            M = D
    return M

def level(alpha, n, m):
    return sqrt(-1/2 * log(alpha/2)) * sqrt((n + m)/(n*m))

max1 = oneside(data1, data2)
max2 = oneside(data2, data1)

print('max1 = %f, max2 = %f' % (max1, max2))

D = max(max1, max2)
l = level(1e-5, len(data1), len(data2))

print('D = %f, l = %f (alpha = %g)' % (D, l, 1e-5))
