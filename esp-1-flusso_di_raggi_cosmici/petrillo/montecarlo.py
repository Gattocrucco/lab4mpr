import numpy as np
from scipy import stats, optimize
import sympy
import uncertainties as un

class Scint(object):
    
    def _asufloat(self, x):
        if isinstance(x, un.UFloat):
            return x
        else:
            return un.ufloat(x, 0)
    
    def __init__(self, long_side_length=480, short_side_length=400, center_depth=0, long_side_offset=0, short_side_offset=0, long_side_inclination=0, short_side_inclination=0, efficiency=1.0, thickness=10):
        
        self._Lx = self._asufloat(short_side_length) / 1000
        self._Ly = self._asufloat(long_side_length) / 1000
        self._z = -self._asufloat(center_depth) / 1000
        self._x = self._asufloat(short_side_offset) / 1000
        self._y = self._asufloat(long_side_offset) / 1000
        self._alpha = self._asufloat(short_side_inclination) * np.pi / 180
        self._beta = self._asufloat(long_side_inclination) * np.pi / 180
        self._efficiency = self._asufloat(efficiency)
        self._thickness = self._asufloat(thickness) / 1000
        
        self._compute_geometry(randomize=False)
    
    @property
    def thickness(self):
        return self._thickness.n
    
    def _urandom(self, x, size=None):
        return stats.norm.rvs(loc=x.n, scale=x.s, size=size)
    
    def sample_geometry(self, size):
        size = int(size)
        assert(1 <= size <= 1e5)
        self._random_Lx = self._urandom(self._Lx, size=size)
        self._random_Ly = self._urandom(self._Ly, size=size)
        self._random_z = self._urandom(self._z, size=size)
        self._random_x = self._urandom(self._x, size=size)
        self._random_y = self._urandom(self._y, size=size)
        self._random_alpha = self._urandom(self._alpha, size=size)
        self._random_beta = self._urandom(self._beta, size=size)
    
    def _compute_geometry(self, randomize=False):
        if randomize:
            Lx    = self._random_Lx
            Ly    = self._random_Ly
            z     = self._random_z
            x     = self._random_x
            y     = self._random_y
            alpha = self._random_alpha
            beta  = self._random_beta
        else:
            Lx    = np.array([self._Lx.n   ])
            Ly    = np.array([self._Ly.n   ])
            z     = np.array([self._z.n    ])
            x     = np.array([self._x.n    ])
            y     = np.array([self._y.n    ])
            alpha = np.array([self._alpha.n])
            beta  = np.array([self._beta.n ])
        
        sa = np.sin(alpha)
        ca = np.cos(alpha)
        sb = np.sin(beta)
        cb = np.cos(beta)
        
        self._Vx, self._Vy, self._P = np.empty((3, 3, len(Lx)))
        
        self._Vx[0] = ca
        self._Vx[1] = 0
        self._Vx[2] = sa
        
        self._Vy[0] = sa * sb
        self._Vy[1] = cb
        self._Vy[2] = -ca * sb
        
        self._P[0] = x
        self._P[1] = y - self._Vy[1] * Ly
        self._P[2] = -z - Lx/2 * self._Vx[2]
        
        self._size = np.array([Lx, Ly])
            
    def _compute_efficiency(self, randomize=False):
        if randomize:
            e = -1
            while not (0 <= e <= 1):
                e = self._urandom(self._efficiency)
            return e
        else:
            return self._efficiency.n
    
    def _make_sympy_solver(self, Vx=None, Vy=None):
        v = sympy.symbols('v:3')
        Pmp = sympy.symbols('Pmp:3')
        args = v
        if Vx is None:
            Vx = sympy.symbols('Vx:3')
            args += Vx
        if Vy is None:
            Vy = sympy.symbols('Vy:3')
            args += Vy
        args += Pmp
    
        S = sympy.Matrix([v, [-vx for vx in Vx], [-vy for vy in Vy], Pmp]).transpose()
        ts = sympy.symbols('t tx ty')
        sol = sympy.solve_linear_system(S, *ts)
        ftx = sympy.lambdify(args, sol[ts[1]])
        fty = sympy.lambdify(args, sol[ts[2]])
    
        return ftx, fty
    
    def _angle(self, v):
        # stub implementation
        return v[2]
    
    # axis order:
    # 0 xyz, 1 mc, 2 geom
    def within(self, v, p, randgeom=False, randeff=False, cachegeom=False, angle=False):
        self._compute_geometry(randomize=randgeom)
        
        if not randgeom and cachegeom and not hasattr(self, '_cache_ftx'):
            self._cache_ftx, self._cache_fty = self._make_sympy_solver(Vx=self._Vx, Vy=self._Vy)
        elif not hasattr(Scint, '_ftx'):
            Scint._ftx, Scint._fty = self._make_sympy_solver()
        
        args = tuple(v.reshape(3,-1,1))
        if not randgeom and hasattr(self, '_cache_ftx'):
            ftx = self._cache_ftx
            fty = self._cache_fty
        else:
            ftx = Scint._ftx
            fty = Scint._fty
            args += tuple(self._Vx.reshape(3,1,-1))
            args += tuple(self._Vy.reshape(3,1,-1))
        args += tuple(self._P.reshape(3,1,-1) - p)
        
        tx = ftx(*args)
        ty = fty(*args)
        
        efficiency = self._compute_efficiency(randomize=randeff)
        
        rt = (np.logical_and(np.logical_and(0 <= tx, tx <= self._size[0].reshape(1,-1)), np.logical_and(0 <= ty, ty <= self._size[1].reshape(1,-1))), efficiency)
        if angle:
            rt += (self._angle(v),)
        return rt
    
    def pivot(self, costheta, phi, tx, ty, randgeom=False, randeff=False, cachegeom=False, angle=False):
        sintheta = np.sqrt(1 - costheta ** 2)
        
        v = np.array([sintheta * np.cos(phi), sintheta * np.sin(phi), costheta])
        
        self._compute_geometry(randomize=randgeom)
        
        p = self._P.reshape(3,1,-1) + self._Vx.reshape(3,1,-1) * tx.reshape(1,-1,1) * self._size[0].reshape(1,1,-1) + self._Vy.reshape(3,1,-1) * ty.reshape(1,-1,1) * self._size[1].reshape(1,1,-1)
        
        horizontal_area = self._size[0] * np.sqrt(1 - self._Vx[2]**2) * self._size[1] * np.sqrt(1 - self._Vy[2]**2)
        
        efficiency = self._compute_efficiency(randomize=randeff)
        
        rt = (v, p, horizontal_area, efficiency)
        if angle:
            rt += (self._angle(v),)
        return rt
    
class MC(object):
    """
    Object to draw random samples and compute acceptance ratios.
    
    Parameters
    ----------
    *scints : objects of class Scint
        The scintillator planes to use. The order identify the scint
        inside the MC object, example if you initialise with
            MC(A,B,C)
        then the index of A is 0, B is 1, C is 2.
    
    Methods
    -------
    random_ray : draw random rays
    run : perform computation for each ray
    count : count number of rays satisfying logic expression
    density : return "d" such that the rate per unit horizontal area is
        r_hor = d * (measured rate)
    
    Example
    -------
    Create Scint objects using the pmt function:
    >>> sA = pmt(2)
    >>> sB = pmt(4)
    Initialise a MC object:
    >>> mc = MC(sA, sB)
    Draw 1000 random rays with angular distribution cos^2 theta:
    >>> mc.random_ray(N=1000, distcos=lambda x: x**2)
    Perform computations with the first scint (sA) as pivot:
    >>> mc.run(pivot_scint=0)
    Count rays that make coincidence:
    >>> mc.count(True, True)
    Count rays that hit sA but not sB:
    >>> mc.count(True, False)
    Count rays that hit sA:
    >>> mc.count(True, ...) # --> 1000 if the efficiency of sA is 1
    This thing should give error:
    >>> mc.count(False, True) # --> ERROR
    because scint 0 was fixed as pivot by mc.run, which means it must always be true.
    To compute the previous count, mc.run must be called again:
    >>> mc.run(pivot_scint=1) # pivot scint is scint 1 aka sB
    >>> mc.count(False, True) # --> OK
    """
    
    def __init__(self, *scints):
        self._scints = list(scints)
    
    def _von_neumann(self, pdf, size):
        out = optimize.minimize_scalar(lambda x: -pdf(x), method='bounded', bounds=[0, 1])
        maxpdf = pdf(out.x)
        
        tempt_sam = stats.uniform.rvs(size=size)
        acc_sam = stats.uniform.rvs(size=size, scale=maxpdf)
        samples = tempt_sam[acc_sam < pdf(tempt_sam)]
        n_tempt = size
        
        while len(samples) < size:
            acceptance = len(samples) / n_tempt
            if acceptance == 0:
                acceptance = 3 / n_tempt
            new_tempt = int(np.ceil((size - len(samples)) / acceptance))
            n_tempt += new_tempt
            
            tempt_sam = stats.uniform.rvs(size=new_tempt)
            acc_sam = stats.uniform.rvs(size=new_tempt, scale=maxpdf)
            new_samples = tempt_sam[acc_sam < pdf(tempt_sam)]
            samples = np.concatenate((samples, new_samples))
        
        return samples[:size]
    
    def sample_geometry(self, size):
        for scint in self._scints:
            scint.sample_geometry(size)
    
    def random_samples(self, N=10000):
        """
        Draw random samples and saves them without doing anything.
        Use in conjunction with ray().
        """
        self._samples = stats.uniform.rvs(size=N)
        self._phi = stats.uniform.rvs(size=N, scale=2 * np.pi)
        self._tx = stats.uniform.rvs(size=N)
        self._ty = stats.uniform.rvs(size=N)
    
    def ray(self, stat):
        """
        Apply a statistic to the samples drawn with random_samples()
        to obtain samples with an arbitrary distribution, then
        obtain random rays. The samples from random_samples()
        are uniformly distributed in [0,1].
        
        Parameters
        ----------
        stat : callable
            An ufunction that takes a number in [0,1] and returns
            a number in [0,1]. (ufunction means it works on arrays)
        
        Example
        -------
        Draw 10000 uniform random samples:
        >>> mc.random_samples(10000)
        Use the angular distribution p(cos(theta)) = cos^2(theta),
        which is obtained by a cube root:
        >>> mc.ray(lambda x: np.cbrt(x))
        
        The same could be accomplished with
        >>> mc.random_ray(N=10000, distcos=lambda x: x**2)
        but by separating the samples from the distribution
        the result of the Monte Carlo is continuous with respect
        to the parameters of the distribution.
        """
        self._costheta = stat(self._samples)
    
    def random_ray(self, N=10000, distcos=None):
        """
        Draw random rays and saves them in the MC object.
        
        Parameters
        ----------
        N : integer, 2 <= N <= 1e6
            Number of rays to sample. The range is for memory reasons.
        distcos : None, distribution from scipy.stats or callable
            The probability distribution for the variable x = cos(theta).
            If None, p(x) = x**2 is used. If a distribution from the module
            scipy.stats, random xs are obtained with distcos.rvs(). If callable,
            it is the pdf of x (eventually non-normalised): p(x) \propto distcos(x).
        """
        N = int(N)
        if not (2 <= N <= 1000000):
            raise ValueError("number %d of samples out of range 2--1000000." % N)
        if distcos is None:
            self._costheta = np.cbrt(stats.uniform.rvs(size=N))
        elif hasattr(distcos, 'rvs'):
            self._costheta = distcos.rvs(size=N)
            if np.any(self._costheta < 0) or np.any(self._costheta > 1):
                raise ValueError('the given distribution must have support in [0,1].')
        elif hasattr(distcos, '__call__'):
            self._costheta = self._von_neumann(distcos, N)
        else:
            raise ValueError('distcos must be either None, an instance of rv_continuous or a function.')
        self._phi = stats.uniform.rvs(size=N, scale=2 * np.pi)
        self._tx = stats.uniform.rvs(size=N)
        self._ty = stats.uniform.rvs(size=N)
            
    def run(self, pivot_scint=0, spectrum=None, **kw):
        """
        Compute for each ray which scints it hits.
        Since rays are needed, random_ray must have been called at least once.
        
        Parameters
        ----------
        pivot_scint : index
            The index of the pivot scint. The pivot scint is the one
            which is used to actually place in space the random rays,
            that is all random rays hit the pivot scint. So later,
            when counting logical expressions, the pivot scint must
            always be True.
        
        Keyword arguments
        -----------------
        randgeom : boolean or string, default to False
            If True, extract at random the geometrical properties of the scints
            before doing all the computation. The distribution used is gaussian
            with standard deviation as given into the Scints objects. If False
            (default), use nominal values.
        randeff : boolean, default to False
            The same for the efficiency of the Scints.
        """
        scints = self._scints.copy()
        if spectrum is True:
            spectrum = pivot_scint
        spectrum_scint = scints[spectrum] if not spectrum is None else None
        pivot = scints.pop(pivot_scint)
        self._pivot = pivot_scint
        
        if not spectrum_scint is pivot:
            v, p, self._horizontal_area, self._pivot_eff = pivot.pivot(self._costheta, self._phi, self._tx, self._ty, **kw)
        else:
            v, p, self._horizontal_area, self._pivot_eff, self._spectr_ang = pivot.pivot(self._costheta, self._phi, self._tx, self._ty, angle=True, **kw)
            self._spectr_within = True
            self._spectr_thick = pivot.thickness
        self._withins = []
        self._efficiencies = []
        for scint in scints:
            if not spectrum_scint is scint:
                w, e = scint.within(v, p, **kw)
            else:
                w, e, self._spectr_ang = scint.within(v, p, angle=True, **kw)
                self._spectr_within = w
                self._spectr_thick = scint.thickness
            self._withins.append(w)
            self._efficiencies.append(e)
        
        self._N = len(self._costheta)
        self._geom0d = not kw.get('randgeom', False)
    
    @property
    def pivot_horizontal_area(self):
        return self._horizontal_area[0] if self._geom0d else self._horizontal_area
    
    @property
    def number_of_rays(self):
        return self._N
    
    def costheta(self, *exprs):
        # stub implementation, do not consider efficiency
        withins, d1 = self._compute_withins(*exprs)
        rt = []
        for within in withins:
            rt.append(self._spectr_ang[np.logical_and(self._spectr_within, within)])
        return rt if d1 else rt[0]
    
    def energy(self, *exprs, smear=False):
        # in MeV
        costheta = self.costheta(*exprs)
        energy = self._spectr_thick / costheta * 1.5 * 100
        if smear:
            energy += stats.norm.rvs(size=len(energy), loc=0, scale=energy/10)
        return energy
    
    def count(self, *exprs, tags=None):
        """
        Count rays which satisfy given logical expression(s).
        A logical expression is a list of either True, False or None,
        correspoding to the scints in order.
            True = the ray hits the scint;
            False = the ray does not hit the scint;
            None = ignore this scint.
        Example: to count rays that hit scint 0 and scint 1 but not scint 2,
        ignoring what happens at scint 3:
            True, True, False, None.
        
        Synonyms
        --------
        Ellipsis can be used instead of None. Remember that Ellipsis can
        also be written as ...
        Anything that can be cast to bool (excluding None and Ellipsis)
        is valid to specify True or False, e.g. 1 or 0 for quick writing.
        An empty expression means all scints True.
        
        Uncertainty
        -----------
        The returned count is a value with uncertainty (from the uncertainties
        package). The uncertainty is the Monte Carlo standard deviation.
        
        Efficiency
        ----------
        The returned count is effectively a count if the efficiencies of the
        scints are 1. If they are not 1, the simulation of the efficiency is
        implemented as an integral, i.e. a hit is weighted with the efficiency
        instead of simulating if the hit happens or not.
        
        Multiple expressions
        --------------------
        The argument *exprs can be either an expression or a list of expressions.
        If a list of expressions, the values returned are properly correlated.
        This is useful to reduce the Monte Carlo uncertainty when calculating
        e.g. ratios of acceptances.
        
        Examples
        --------
        Calculate the ratio of two acceptances:
        >>> c01, c02 = mc.count([True, True, ...], [True, ..., True])
        >>> r = c01 / c02 # the uncertainty is propagated with correlation
        """
        withins, d1 = self._compute_withins(*exprs)
        
        if self._geom0d:
            withins = withins[:,:,0] # eliminate geometry samples axis
            counts = np.sum(withins, axis=1)
            counts_cov = np.atleast_2d(np.cov(withins, ddof=1)) * withins.shape[1]
            ucounts = np.array(un.correlated_values(counts, counts_cov, tags=tags))
            return ucounts if d1 else ucounts[0]
        else:
            counts = np.sum(withins, axis=1)
            return counts if d1 else counts[0]
    
    def _compute_withins(self, *exprs):
        exprs = list(exprs)
        if len(exprs) == 0:
            exprs = [True]
        if not any([hasattr(expr, '__len__') for expr in exprs]):
            exprs = [exprs]
            d1 = False
        else:
            d1 = True
        for i in range(len(exprs)):
            expr = exprs[i]
            if not hasattr(expr, '__len__'):
                expr = [expr]
            if len(expr) == 0:
                expr = [True]
            expr = list(expr)
            if len(expr) == 1:
                expr *= len(self._scints)
            if len(self._scints) != len(expr):
                raise ValueError("exprs[%d] has length %d != %d" % (i, len(expr), len(self._scints)))
            pe = expr[self._pivot]
            if pe is None or pe is Ellipsis or not pe:
                raise ValueError("exprs[%d] has pivot scint (%d) = %s != True" % (i, self._pivot, str(expr[self._pivot])))
            expr.pop(self._pivot)
            exprs[i] = expr
                
        withins = np.ones((len(exprs),) + self._withins[0].shape) * self._pivot_eff
        for j in range(len(exprs)):
            expr = exprs[j]
            for i in range(len(self._withins)):
                if expr[i] is None or expr[i] is Ellipsis:
                    continue
                elif expr[i]:
                    w = self._withins[i] * self._efficiencies[i]
                else:
                    w = 1 - self._withins[i] * self._efficiencies[i]
                withins[j] *= w
        
        return withins, d1
        
    def density(self, *exprs):
        """
        Let R be the rate of hits for a given logical expression,
        and R_hor the rate per unit area, this method compute the
        factor D such that
            R_hor = D * R.
        See the docstring of MC.count() for how to specify logical expressions
        in the *exprs argument.
        
        Area
        ----
        The horizontal area is obtained projecting the pivot scint.
        
        Uncertainties
        -------------
        The uncertainties are Monte Carlo standard deviations. No uncertainty
        is accounted for the horizontal area, but if randgeom=True has been
        specified when calling MC.run(), the horizontal area of the pivot
        scint has been randomized.
        
        Multiple expressions
        --------------------
        Multiple expressions are treated as in MC.count(), properly
        storing correlations in the results.
        """
        count = self.count(*exprs)
        return self.number_of_rays / (count * ha)
    
    def long_run(self, *expr, **kw):
        import lab
        N = kw.pop('N', 1e8)
        N = int(N)
        times = N // 1000000
        rem = N % 1000000
        ns = times * [1000000] + ([rem] if rem != 0 else [])
        
        count = 0
        eta = lab.Eta()
        s = 0
        for n in ns:
            self.random_ray(N=n)
            self.run(randeff=False, randgeom=False, cachegeom=True)
            count += self.count(*expr)
            s += n
            eta.etaprint(s / N)
        
        if self._geom0d:
            ha = self._horizontal_area[0]
        elif len(count.shape) == 2:
            ha = self._horizontal_area.reshape(1,-1)
        else:
            ha = self._horizontal_area
        return N / (count * ha)

def pmt(idx, efficiency=1.0):
    """
    Costruisce l'oggetto Scint per uno dei nostri PMT,
    con i numeri che usiamo a lab, mettendo tutte le caratteristiche
    geometriche misurate con le incertezze. L'efficienza si
    specifica con l'argomento (che può avere incertezza).
    
    Esempio:
        s = pmt(1, efficiency=ufloat(0.95, 0.01))
    crea l'oggetto che rappresenta il PMT 1 con efficienza 95 % +- 1 %.
    """
    if not (1 <= idx <= 6):
        raise ValueError("PMT index %d out of range 1--6." % idx)
    
    long_side_angles = np.array([0.2, 0.3, 0.4, 0.3, 0.6, 0.4])
    short_side_angles = np.array([0.7, 1.0, 1.2, 0.8, 0.2, 0.8])

    long_side_angles -= np.mean(long_side_angles)
    short_side_angles -= np.mean(short_side_angles)

    long_side_inclinations = []
    short_side_inclinations = []
    for beta, alpha in zip(long_side_angles, short_side_angles):
        long_side_inclinations.append(un.ufloat(beta, 0.05))
        short_side_inclinations.append(un.ufloat(alpha, 0.05))

    long_side_lengths = [482, 482, 481, 480, 481, 480]
    short_side_lengths = [400, 404, 398, 400, 395, 398]

    for i in range(len(long_side_lengths)):
        long_side_lengths[i] = un.ufloat(long_side_lengths[i], 1)
        short_side_lengths[i] = un.ufloat(short_side_lengths[i], 2)

    center_depths = [0, 102, 205, 308, 411, 804]

    for i in range(len(center_depths)):
        center_depths[i] = un.ufloat(center_depths[i], 1)
    center_depths[0].std_dev = 0
    center_depths[-1].std_dev = 2

    short_side_offsets = [0, -5, 1, 1, 2, 0]
    short_side_offsets_un = [0, 1, 1, 1.4, 1.4, 2]
    long_side_offsets = [543, 540, 536, 539, 531]
    s = np.std(long_side_offsets, ddof=1)
    long_side_offsets.append(np.mean(long_side_offsets))

    for i in range(len(short_side_offsets)):
        short_side_offsets[i] = un.ufloat(short_side_offsets[i], short_side_offsets_un[i])
        long_side_offsets[i] = un.ufloat(long_side_offsets[i], 1)
    long_side_offsets[-1].std_dev = s
    
    i = 6 - idx
    
    return Scint(long_side_length=long_side_lengths[i], short_side_length=short_side_lengths[i], long_side_inclination=long_side_inclinations[i], short_side_inclination=short_side_inclinations[i], long_side_offset=long_side_offsets[i], short_side_offset=short_side_offsets[i], center_depth=center_depths[i], efficiency=efficiency)
