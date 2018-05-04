import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
import sys
from scipy import optimize, stats
import collections
import uncertainties as un
from uncertainties import unumpy as unp

try:
    import numba
except ImportError:
    def numba_jit(*args, **kwargs):
        def ret_fun(fun):
            return fun
        return ret_fun
    have_numba = False
else:
    numba_jit = numba.jit
    have_numba = True

def interp(x, y):
    """
    Interpolate linearly y(x).
    
    Parameters
    ----------
    x : array
        Must be sorted.
    y : array
    
    Returns
    -------
    fun : compiled f8->f8 function
        Interpolating function.
    """
    @numba_jit('f8(f8)', nopython=True)
    def fun(x0):
        assert x[0] <= x0 <= x[-1]
        idx = np.searchsorted(x, x0)
        # we have x[idx-1] < x0 <= x[idx]
        if idx == 0:
            idx = 1
        center_x = x[idx - 1]
        center_y = y[idx - 1]
        slope = (y[idx] - center_y) / (x[idx] - center_x)
        return slope * (x0 - center_x) + center_y
    
    return fun

def histogram(a, bins=10, range=None, weights=None, density=None):
    """
    Same as numpy.histogram, but returns uncertainties.
    The uncertainty of a bin with density=False is:
    Case 1 (no weights):
        Square root of the count.
    Case 2 (weights):
        Quadrature sum of the weights (the same as case 1 if the weights are unitary).
    If density=True, the uncertainty is simply divided by the same factor
    as the counts.
    
    Returns
    -------
    hist
    bin_edges
    unc_hist :
        Uncertainty of hist.
    """
    hist, bin_edges = np.histogram(a, bins=bins, range=range, weights=weights, density=density)
    
    if weights is None and not density:
        unc_hist = np.where(hist > 0, np.sqrt(hist), 1)
    elif weights is None:
        counts, _ = np.histogram(a, bins=bins, range=range)
        unc_hist = np.where(counts > 0, np.sqrt(counts), 1) / (len(a) * np.diff(bin_edges))
    else:
        unc_hist, _ = np.histogram(a, bins=bins, range=range, weights=weights ** 2)
        unc_hist = np.where(unc_hist > 0, np.sqrt(unc_hist), 1)
        if density:
            unc_hist /= (np.sum(weights) * np.diff(bin_edges))
    
    return hist, bin_edges, unc_hist

def bar(edges, counts, ax=None, **kwargs):
    """
    Draw histogram with 'step' linestyle, given edges and bins.
    
    Parameters
    ----------
    edges, counts :
        As returned e.g. by numpy.histogram.
    ax : None or matplotlib axis
        If None use current axis, else axis given.
    
    Keyword arguments
    -----------------
    Passed to ax.plot.
    
    Returns
    -------
    Return value from ax.plot.
    """
    dup_edges = np.concatenate([[edges[0]], edges, [edges[-1]]])
    dup_counts = np.concatenate([[0], [counts[0]], counts, [0]])
    if ax is None:
        ax = plt.gca()
    kwargs.update(drawstyle='steps')
    return ax.plot(dup_edges, dup_counts, **kwargs)
    
def rebin(a, n):
    """
    Sum a in adiacent groups of n elements. If the length of a is not a
    multiple of n, last elements are dropped.
    
    Parameters
    ----------
    a : array
    n : integer
    
    Returns
    -------
    out : array
        The length is len(a) // n.
    """
    assert n == int(n)
    n = int(n)
    out = np.zeros(len(a) // n)
    for i in range(n):
        out += a[i::n][:len(out)]
    return out

def errorbar(ux, uy, *args, **kwargs):
    x = unp.nominal_values(ux)
    y = unp.nominal_values(uy)
    xerr = unp.std_devs(ux)
    yerr = unp.std_devs(uy)
    if np.any(xerr != 0):
        kwargs.update(xerr=xerr)
    if np.any(yerr != 0):
        kwargs.update(yerr=yerr)
    ax = kwargs.pop('ax', plt)
    return ax.errorbar(x, y, *args, **kwargs)

def clear_lines(nlines, nrows):
    """
    Clear the last nlines terminal lines up to row nrows.
    """
    for i in range(nlines):
        sys.stdout.write('\033[F\r%s\r' % (" " * nrows,))
    sys.stdout.flush()

def make_von_neumann(density, domain, max_cycles=100000):
    """
    density = positive function float -> float
    domain = [a, b] domain of density
    """
    out = optimize.minimize_scalar(lambda x: -density(x), bounds=domain, method='bounded')
    if not out.success:
        raise RuntimeError('cannot find the maximum of density in [%g, %g]' % domain)
    top = density(out.x)
    left = domain[0]
    right = domain[1]
    
    if have_numba and not isinstance(density, numba.targets.registry.CPUDispatcher):
        density = numba.jit('f8(f8)', nopython=True)(density)
    
    @numba_jit('f8', nopython=True)
    def von_neumann():
        i = 0
        while i < max_cycles:
            candidate = np.random.uniform(left, right)
            height = np.random.uniform(0, top)
            if height <= density(candidate):
                return candidate
            i += 1
        return np.nan
    
    return von_neumann

@numba_jit('f8[3](f8[3],f8[3])', nopython=True, cache=True)
def cross(a, b):
    """
    cross product a x b
    """
    return np.array([
        a[1]*b[2] - a[2]*b[1],
        a[2]*b[0] - a[0]*b[2],
        a[0]*b[1] - a[1]*b[0]
    ])

@numba_jit('f8[3](f8[3])', nopython=True, cache=True)
def versor(a):
    """
    normalize a (returns a / |a|)
    """
    return a / np.sqrt(np.sum(a ** 2))

def errorsummary(x):
    """
    Returns error components of a ufloat as a ordered dictionary
    where the keys are the tags of the components. Components
    with the same tag are summed over. The ordering is greater
    component first.
    """
    comps = x.error_components()
    
    tags = set(map(lambda v: v.tag, comps.keys()))
    var = dict(zip(tags, [0] * len(tags)))
    
    for (v, sd) in comps.items():
        var[v.tag] += sd ** 2
    
    tags = list(tags)
    sds = np.sqrt(np.array([var[tag] for tag in tags]))
    idx = np.argsort(sds)[::-1]
    d = collections.OrderedDict()
    for i in idx:
        d[tags[i]] = sds[i]
    
    return d

def weighted_mean(y):
    """
    Weighted mean (with covariance matrix).
    
    Parameters
    ----------
    y : array of ufloats
    
    Returns
    -------
    a : ufloat
        Weighted mean of y.
    Q : float
        Chisquare (the value of the minimized quadratic form at the minimum).
    """
    inv_covy = np.linalg.inv(un.covariance_matrix(y))
    vara = 1 / np.sum(inv_covy)
    a = vara * np.sum(np.dot(inv_covy, y))
    assert np.allclose(vara, a.s ** 2)
    
    res = unp.nominal_values(y) - a.n
    Q = float(res.reshape(1,-1) @ inv_covy @ res.reshape(-1,1))
    
    return a, Q

@numba_jit('u4(f8,f8[:],f8[:],f8[:],f8)', nopython=True, cache=True)
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

def credible_interval(samples, cl=0.68, ax=None):
    kde = stats.gaussian_kde(samples)
    pdf = kde(samples)
    idx = np.argsort(pdf)
    sorted_samples = samples[idx]
    interval_samples = sorted_samples[-int(np.round(cl * len(samples))):]
    left = np.min(interval_samples)
    right = np.max(interval_samples)
    act_cl = len(interval_samples) / len(samples)
    out = optimize.minimize_scalar(lambda x: -kde(x), bracket=(left, right))
    if not out.success:
        raise RuntimeError('can not find mode of pdf')
    if not (ax is None):
        ax.plot(samples, pdf, '.k', label='samples')
        l = ax.get_ylim()
        ax.plot(2 * [out.x[0]], l, '--k', scaley=False, label='mode')
        rect = patches.Rectangle(
            (left, l[0]),
            right - left,
            l[1] - l[0],
            facecolor='lightgray',
            edgecolor='none',
            zorder=-1,
            label='%.3f CR' % act_cl
        )
        ax.add_patch(rect)
        ax.legend(loc='best', fontsize='small')
        ax.grid(linestyle=':')
        ax.set_xlabel('value')
        ax.set_ylabel('pdf')
    return out.x[0], left, right, act_cl

def loadtxt(fname, dtype=float, usecols=None, unpack=False):
    """
    Implements a subset of numpy.loadtxt's functionality,
    using pandas.read_csv (which is much faster).
    """
    import pandas
    if usecols is None:
        guess = pandas.read_csv(fname, header=0, nrows=2, delim_whitespace=True)
        ncolumns = guess.shape[1]
    else:
        ncolumns = len(usecols)
    names = list(map(str, range(ncolumns)))
    dataframe = pandas.read_csv(fname, delim_whitespace=True, usecols=usecols, dtype=dtype, header=None, names=names)
    array = dataframe.values
    if unpack:
        array = array.T
    return array
