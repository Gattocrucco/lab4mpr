from pylab import *
from scipy.special import expi, gamma

gamma_mascheroni = 0.57721566490153286060

var_eff = lambda mu: (expi(mu) - log(mu) - gamma_mascheroni) / (exp(mu) - 1)

mus = linspace(0, 700, 1000)[1:]

exact = var_eff(mus)

app_part = [gamma(1 + k) / mus**(1 + k) for k in range(8)]

apps = cumsum(app_part, axis=0)

figure('test eff')
clf()

subplot(211)
plot(mus, exact, label='exact')
for i in range(len(apps)):
    plot(mus, apps[i], label='approx %d' % (i + 1))
legend(loc=0, fontsize='small')
yscale('log')

subplot(212)
for i in range(len(apps)):
    plot(mus, abs(apps[i] - exact) / exact, label='|(a-e)/e|, %d' % (i + 1))
legend(loc=0, fontsize='small')
yscale('log')

show()
