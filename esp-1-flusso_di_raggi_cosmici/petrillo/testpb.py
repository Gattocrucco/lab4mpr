from pylab import *
import numba
import uncertainties as un

print('loading data...')
data1 = loadtxt('../de0_data/piombo_con_tutto_c12.dat', unpack=True, usecols=(1,))
data2 = loadtxt('../de0_data/piombo_senza_c12.dat', unpack=True, usecols=(1,))

def mode(data):
    counts, bins = histogram(data, bins='sqrt')
    m_idx = argmax(counts)
    m = (bins[m_idx+1] + bins[m_idx]) / 2
    m_unc = (bins[m_idx+1] - bins[m_idx]) / 2
    return un.ufloat(m, m_unc)
    
def average(data):
    return un.ufloat(mean(data), std(data, ddof=1) / sqrt(len(data)))

m1 = mode(data1)
m2 = mode(data2)

mu1 = average(data1)
mu2 = average(data2)

print('piombo: moda = {}\nsenza piombo: moda = {}'.format(m1, m2))
print('piombo: media = {}\nsenza piombo: media = {}'.format(mu1, mu2))

data1 = sort(data1)
data2 = sort(data2)

@numba.jit(cache=True, nopython=True)
def oneside(dataa, datab):
    # dataa and datab must be sorted
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
D = max(max1, max2)
alpha = 1e-7
l = level(alpha, len(data1), len(data2))

print('D = %f, l = %f (alpha = %g)' % (D, l, alpha))


