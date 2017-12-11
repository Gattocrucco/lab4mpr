from pylab import *
from scipy.stats import poisson, norm
from scipy.special import xlogy
from scipy.optimize import bisect

# log likelihood ratio
# (!) k >= 0, mu >= 0
poisson_order = lambda k, mu: k + xlogy(k, mu) - xlogy(k, k)

# band vertical line
# prefer low k
# (!) mu >= 0, CL > 0, CL < 1
def coverage(mu, CL):
    mus = array([floor(mu), ceil(mu)], dtype='int32')
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

# fast but tight right bound to the band
# (!) k >= 0, CL > 0, CL < 1
# norm.ppf: bound for large k
# -log(1-CL): bound for k = 0 for big CL
# exp(-1): bound for k = 0 for small CL
def right_bound(k, CL):
    return k + ceil(sqrt(k) * norm.ppf((1+CL)/2)) - log(1 - CL) + exp(-1)

# band horizontal line
# (!) k >= 0, CL > 0, CL < 1
def interval(k, CL):
    f = lambda mu: coverage(mu, CL)[1] - k + 1/2
    mu_left = bisect(f, 0, k, rtol=1e-5) if k > 0 else 0
    f = lambda mu: coverage(mu, CL)[0] - k - 1/2
    mu_right = bisect(f, k, right_bound(k, CL), rtol=1e-5)
    return array([mu_left, mu_right])

if __name__ == '__main__':
    CLS = [0.90, 0.95, 0.99]
    mus = linspace(0, 20, 500)

    figure('FC poisson band').set_tight_layout(True)
    clf()

    for CL in CLS:
        ks = empty((len(mus), 2))
        for i in range(len(mus)):
            ks[i] = coverage(mus[i], CL)
        p = plot(mus, ks[:,0], '-', label='CL = %.4f' % CL)
        plot(mus, ks[:,1], '-', color=p[0].get_color())
        uk = unique(ks[:,0])
        ints = array([interval(k, CL) for k in uk])
        plot(ints[:,0], uk, '<', color=p[0].get_color())
        plot(ints[:,1], uk, '>', color=p[0].get_color())

    title('F.C. poisson confidence band')
    xlabel('$\\mu$')
    ylabel('k')
    legend(loc='best', fontsize='small')

    show()
