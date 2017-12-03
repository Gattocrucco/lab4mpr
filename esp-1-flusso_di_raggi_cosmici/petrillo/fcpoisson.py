from pylab import *
from scipy.stats import poisson
from scipy.special import xlogy

# log likelihood ratio
# (!) k >= 0, mu >= 0
poisson_order = lambda k, mu: k + xlogy(k, mu) - xlogy(k, k)

# band vertical line
# prefer low k
# (!) mu >= 0, CL > 0, CL < 1
def coverage(mu, CL):
    mus = array([floor(mu), ceil(mu)])
    startk = mus[argmax(poisson_order(mus, mu))]
    left = startk
    right = startk
    coverage = poisson.pmf(startk, mu)
    while coverage < CL:
        if left > 0:
            opts = array([left - 1, right + 1])
            i = argmax(poisson_order(opts, mu))
            if i == 0:
                left -= 1
            else:
                right += 1
            coverage += poisson.pmf(opts[i], mu)
        else:
            right += 1
            coverage += poisson.pmf(right, mu)
    return array([left, right])

CLS = [0.68, 0.90, 0.95, 0.99]
mus = linspace(0, 10, 1000)

figure('FC poisson band').set_tight_layout(True)
clf()

for CL in CLS:
    ks = empty((len(mus), 2))
    for i in range(len(mus)):
        ks[i] = coverage(mus[i], CL)
    p = plot(mus, ks[:,0], '-', label='CL = %.2f' % CL)
    plot(mus, ks[:,1], '-', color=p[0].get_color())

title('F.C. poisson confidence band')
xlabel('$\\mu$')
ylabel('k')
legend(loc='best', fontsize='small')

show()