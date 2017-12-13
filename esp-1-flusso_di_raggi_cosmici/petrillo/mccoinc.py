from pylab import *
from scipy import stats
from lab import xe

def extract_seq(r, T, tm):
    n = int(T * r)
    ti = stats.expon.rvs(size=n, scale=1/r)
    s = sum(ti)
    while s < T:
        n = int(floor((T - s) * r) + 1)
        new_ti = stats.expon.rvs(size=n, scale=1/r)
        s += sum(new_ti)
        ti = concatenate((ti, new_ti))
    
    z = diff(ti < tm) > 0
    while any(z):
        ti[:-1] += ti[1:] * z
        ti = ti[1:][logical_not(z)]
        z = diff(ti < tm) > 0
    
    ti = cumsum(ti)
    return ti[ti <= T]

def count_coinc(ti1, ti2, tc1, tc2):
    t = concatenate((ti1, ti2))
    tc = concatenate((ones(ti1.shape) * tc1, ones(ti2.shape) * tc2))
    tx = concatenate((zeros(ti1.shape), ones(ti2.shape)))
    
    i = argsort(t)
    t = diff(t[i])
    tc = tc[i]
    tx = diff(tx[i])
    
    r = logical_and(t < tc[:-1], tx != 0)
    
    return sum(r), std(r, ddof=1) * sqrt(len(r))

if __name__ == '__main__':
    r1 = 50e3
    r2 = 16e3
    T = 1000
    assert(max(T*r1, T*r2) <= 1e8)
    tm = 700e-9
    tc1 = 292e-9
    tc2 = 184e-9
    tm = max([tm, tc1, tc2])
    
    t1 = extract_seq(r1, T, tm)
    t2 = extract_seq(r2, T, tm)
    
    c, dc = count_coinc(t1, t2, tc1, tc2)
    
    c_1 = r1 * (1 - tm * r1) * r2 * (1 - tm * r2) * (tc1 + tc2) * T
    
    print('MC  = %12s' % xe(c, sqrt(c + dc**2)))
    print('att = %12s' % xe(c_1, sqrt(c_1)))
