from pylab import *

# Verifica che, tenendo conto della massa del nucleo,
# la sezione d'urto differenziale cambia al secondo
# ordine nel rapporto delle masse.

figure('testrf')
clf()

def domega_fact(x, phi):
    return (1 - phi**2 * (1 - 2 * x**2)) / sqrt(1 - phi**2 * (1 - x**2)) + 2 * phi * x

def denom_fact(x, phi):
    return (1/(1+phi) * (1 - (sqrt(1-phi**2 * (1-x**2)) + phi*x) * x + phi) / (1 - x)) ** 2

x = linspace(0.001, 0.999, 500)
phis = [1/10, 1/5, 1/2]

for phi in phis:
    line, = plot(x, denom_fact(x, phi), label=str(phi))
    plot(x, domega_fact(x, phi), color=line.get_color())

legend(loc=0)

show()
