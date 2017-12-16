from pylab import *
import numpy as np
from scipy import stats
from lab import xe
import numba

@numba.jit(numba.uint32(numba.float64, numba.float64[:], numba.float64[:], numba.float64[:], numba.float64), nopython=True, cache=True)
def _coinc(T, r, tc, tm, tand):
    """
    T = total time
    r = rates (!) > 0
    tc = durations (!) > 0
    tm = dead times (!) >= tc
    tand = superposition time for coincidence (!) > 0
    """
    tau = 1 / r
    n = len(tau)
    
    ncoinc = 0
    
    # generate an event on each sequence
    t = -np.ones(n) * tm
    for i in range(n):
        nt = 0
        while nt < tm[i]:
            nt += np.random.exponential(scale=tau[i])
        t[i] += nt
    
    # first sequence for which a new event is generated after a coincidence
    first = 0
        
    # check if total time elapsed
    while t[first] < T:
        
        # intersection interval of events
        il = t[first]
        ir = t[first] + tc[first]
        
        # minimum of right endpoints in case of coincidence
        rmin = ir
        rmini = first
        
        # iterate over sequences
        for i in range(n):
            if i != first:
                coinc_found = False
            
                # check if coincidence is still possible
                while t[i] < ir - tand:
                
                    # check for coincidence
                    nil = max(il, t[i])
                    nir = min(ir, t[i] + tc[i])
                    if nir - nil >= tand:
                        il = nil
                        ir = nir
                        coinc_found = True
                        if t[i] + tc[i] < rmin:
                            rmin = t[i] + tc[i]
                            rmini = i
                        break
                    
                    # generate a new event
                    nt = 0
                    while nt < tm[i]:
                        nt += np.random.exponential(scale=tau[i])
                    t[i] += nt
                
                if not coinc_found:
                    break
                
        if coinc_found and il < T:
            ncoinc += 1
            first = rmini
        
        # generate a new event on the first sequence
        nt = 0
        while nt < tm[first]:
            nt += np.random.exponential(scale=tau[first])
        t[first] += nt
    
    return ncoinc

def coinc(T, tand, *seqs):
    """
    Simulate logical signals and count coincidences.
    
    Arguments
    ---------
    T : number >= 0
        Total time.
    tand : number >= 0
        Minimum superposition time to yield a coincidence.
    *seqs : r1, tc1, tm1, r2, tc2, tm2, ...
        r = Rate of signals.
        tc = Duration of signal.
        tm = Non restartable dead-time. If tm < tc, tc is used instead.
    
    Returns
    -------
    N : integer
        Number of coincidences. Since it is a count, an estimate of the variance is N itself.
    """
    T = np.float64(T)
    if T < 0:
        raise ValueError('Total time is negative.')
    
    tand = np.float64(tand)
    if tand < 0:
        raise ValueError('Superposition time is negative.')
    
    if len(seqs) % 3 != 0:
        raise ValueError('Length of seqs is not a multiple of 3.')
    if len(seqs) / 3 < 2:
        raise ValueError('There are less than 2 sequences in seqs.')
    
    seqs = np.array(seqs, dtype=np.float64)
    r = seqs[::3]
    tc = seqs[1::3]
    tm = np.max((seqs[2::3], tc), axis=0)
    
    if not all(r > 0):
        ValueError('All rates must be positive.')
    if not all(tc > 0):
        ValueError('All durations must be positive.')
    
    return _coinc(T, r, tc, tm, tand)

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
    r1 = 5.32e3
    r2 = 2.61e3
    T = 100
    assert(max(T*r1, T*r2) <= 1e8)
    tm = 700e-9
    tc1 = 254e-9
    tc2 = 270e-9
    tm = max([tm, tc1, tc2])
    
    t1 = extract_seq(r1, T, tm)
    t2 = extract_seq(r2, T, tm)
    
    c, dc = count_coinc(t1, t2, tc1, tc2)
    
    c_1 = r1 * (1 - tm * r1) * r2 * (1 - tm * r2) * (tc1 + tc2) * T
    
    print('MC  = %12s' % xe(c, dc))
    print('att = %d' % c_1)
