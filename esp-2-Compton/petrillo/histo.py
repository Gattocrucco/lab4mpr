from pylab import *
import sys

data = loadtxt(sys.argv[1], unpack=True)

logscale = 'log' in sys.argv[1:]

def partial_sum(a, n):
    out = zeros(len(a) // n)
    for i in range(n):
        out += a[i::n][:len(out)]
    return out

histo = partial_sum(data, 8)

figure('histo')
clf()
bar(arange(len(histo)), histo, width=1, log=logscale)
show()
