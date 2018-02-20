from pylab import *
import sys

data = loadtxt(sys.argv[1], unpack=True)

logscale = 'log' in sys.argv[1:]

cut = 8

def partial_sum(a, n):
    out = zeros(len(a) // n)
    for i in range(n):
        out += a[i::n][:len(out)]
    return out

histo = partial_sum(data, cut)

figure('histo')
clf()
bar(arange(len(histo)) * cut, histo, width=cut, log=logscale)
show()
