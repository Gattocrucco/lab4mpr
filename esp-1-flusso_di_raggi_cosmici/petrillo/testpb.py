from pylab import *
import numba
import uncertainties as un

print('loading data...')
data1 = loadtxt('../de0_data/piombo_con_tutto_c12.dat', unpack=True, usecols=(1,))
data2 = loadtxt('../de0_data/piombo_senza_c12.dat', unpack=True, usecols=(1,))

############ MEAN AND MODE ############

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

############ PLOT ############

#nbins = int(sqrt(min(len(data2), len(data1))))
nbins = 500
# crea istogrammi
MIN = -3.3/8192
MAX = 5999*3.3/8192

counts1, edges = histogram(data1, bins=nbins, range=(MIN, MAX))
counts2, edges = histogram(data2, bins=nbins, range=(MIN, MAX))
counts1 = counts1 / len(data1) / (MAX - MIN)
counts2 = counts2 / len(data2) / (MAX - MIN)

figure('piombo energia').set_tight_layout(False)
clf()
xlims = (-.01,1)

# plot istogrammi insieme
subplot(211)
bar(edges[:-1], counts2, log=False, align='edge', width=(MAX-MIN)/nbins, color='gray', label='senza piombo')
dup_edges = empty(2 * len(edges))
dup_edges[2 * arange(len(edges))] = edges
dup_edges[2 * arange(len(edges)) + 1] = edges
dup_counts = zeros(2 * len(edges))
dup_counts[2 * arange(len(edges) - 1) + 1] = counts1
dup_counts[2 * arange(len(edges) - 1) + 2] = counts1
plot(dup_edges, dup_counts, '-k', label='con piombo', linewidth=1)
ylabel('densità [arb.un.$^{-1}$]')
legend(loc=0, fontsize='small')
xticks([])
xlim(*xlims)

# plot differenza istogrammi
subplot(212)
# @numba.jit(numba.float64[:](numba.float64[:], numba.float64[:], numba.float64[:]), cache=True, nopython=True)
# def diff_cumulant(x, data1, data2):
#     rt = empty(len(x))
#     for i in range(len(x)):
#         rt[i] = sum(data1 < x[i]) / len(data1) - sum(data2 < x[i]) / len(data2)
#     return rt
dup_counts[:] = 0
dup_counts[2 * arange(len(edges) - 1) + 1] = counts1 - counts2
dup_counts[2 * arange(len(edges) - 1) + 2] = counts1 - counts2
plot(dup_edges, dup_counts, '-k', label='piombo $-$ senza piombo', linewidth=1)
centers = edges[:-1] + (edges[1]-edges[0])/2
def error(counts, data):
    return sqrt(counts*len(data) * (MAX - MIN)) / len(data) / (MAX - MIN)
errors = sqrt(error(counts1, data1)**2 + error(counts2, data2)**2)
errorbar(centers, counts1 - counts2, yerr=errors, fmt=',k', capsize=0, ecolor='gray')
legend(loc=0, fontsize='small')
ylabel('densità [arb.un.$^{-1}$]')
xlim(*xlims)

xlabel('energia [arb.un.]')

show()

############ KOLMOGOROV-SMIRNOV TEST ############

data1 = sort(data1)
data2 = sort(data2)

@numba.jit(cache=True, nopython=True)
def oneside(dataa, datab):
    # dataa and datab must be sorted
    M = 0
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
