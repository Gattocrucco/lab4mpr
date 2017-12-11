import numpy as np
from scipy import stats
import sympy
import uncertainties as un
import lab

class Scint(object):
    
    def _asufloat(self, x):
        if isinstance(x, un.core.Variable):
            return x
        else:
            return un.ufloat(x, 0)
    
    def __init__(self, long_side_length=480, short_side_length=400, center_depth=0, short_side_offset=0, long_side_inclination=0, short_side_inclination=0, efficiency=1.0):
        
        self._Lx = self._asufloat(short_side_length) / 1000
        self._Ly = self._asufloat(long_side_length) / 1000
        self._z = -self._asufloat(center_depth) / 1000
        self._x = self._asufloat(short_side_offset) / 1000
        self._alpha = self._asufloat(short_side_inclination) * np.pi / 180
        self._beta = self._asufloat(long_side_inclination) * np.pi / 180
        self._efficiency = self._asufloat(efficiency)
        
        self._compute_geometry(randomize=False)
        
    def _urandom(self, x):
        return stats.norm.rvs(loc=x.n, scale=x.s)
    
    def _compute_geometry(self, randomize=False):
        if randomize:
            Lx = self._urandom(self._Lx)
            Ly = self._urandom(self._Ly)
            z = self._urandom(self._z)
            x = self._urandom(self._x)
            alpha = self._urandom(self._alpha)
            beta = self._urandom(self._beta)
        else:
            Lx = self._Lx.n
            Ly = self._Ly.n
            z = self._z.n
            x = self._x.n
            alpha = self._alpha.n
            beta = self._beta.n
            
        self._Vx = np.array([np.cos(alpha), 0, np.sin(alpha)])
        self._Vy = np.array([np.sin(alpha) * np.sin(beta), np.cos(beta), -np.cos(alpha) * np.sin(beta)])
        self._P = np.array([x, 0, -z - Lx/2 * np.sin(alpha)])
        
        return Lx, Ly, alpha, beta
    
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

    def within(self, v, p, randgeom=False, randeff=False, cachegeom=False):
        Lx, Ly, _, _ = self._compute_geometry(randomize=randgeom)
        
        if not randgeom and cachegeom and not hasattr(self, '_cache_ftx'):
            self._cache_ftx, self._cache_fty = self._make_sympy_solver(Vx=self._Vx, Vy=self._Vy)
        elif not hasattr(Scint, '_ftx'):
            Scint._ftx, Scint._fty = self._make_sympy_solver()
        
        args = tuple(v.reshape(3,-1))
        if not randgeom and hasattr(self, '_cache_ftx'):
            ftx = self._cache_ftx
            fty = self._cache_fty
        else:
            ftx = Scint._ftx
            fty = Scint._fty
            args += tuple(self._Vx.reshape(-1,1))
            args += tuple(self._Vy.reshape(-1,1))
        args += tuple(self._P.reshape(-1,1) - p.reshape(3,-1))
        
        tx = ftx(*args)
        ty = fty(*args)
        
        efficiency = self._compute_efficiency(randomize=randeff)
        
        return np.logical_and(np.logical_and(0 <= tx, tx <= Lx), np.logical_and(0 <= ty, ty <= Ly)), efficiency
    
    def pivot(self, costheta, phi, tx, ty, randgeom=False, randeff=False, cachegeom=False):
        sintheta = np.sqrt(1 - costheta ** 2)
        
        v = np.array([sintheta * np.cos(phi), sintheta * np.sin(phi), costheta])
        
        Lx, Ly, alpha, beta = self._compute_geometry(randomize=randgeom)
        
        p = self._P.reshape(-1,1) + self._Vx.reshape(-1,1) * tx.reshape(1,-1) * Lx + self._Vy.reshape(-1,1) * ty.reshape(1,-1) * Ly
        
        area = Lx * Ly
        horizontal_area = area * np.cos(alpha) * np.cos(beta)
        
        efficiency = self._compute_efficiency(randomize=randeff)
        
        return v, p, horizontal_area, efficiency

class MC(object):
    
    def __init__(self, *scints):
        self._scints = list(scints)
    
    def random_ray(self, N=10000):
        N = int(N)
        if not (2 <= N <= 1000000):
            raise ValueError("number %d of samples out of range 2--1000000." % N)
        self._costheta = np.cbrt(stats.uniform.rvs(size=N))
        self._phi = stats.uniform.rvs(size=N, scale=2 * np.pi)
        self._tx = stats.uniform.rvs(size=N)
        self._ty = stats.uniform.rvs(size=N)
            
    def run(self, pivot_scint=0, **kw):
        scints = self._scints.copy()
        pivot = scints.pop(pivot_scint)
        self._pivot = pivot_scint
        
        v, p, self._horizontal_area, self._pivot_eff = pivot.pivot(self._costheta, self._phi, self._tx, self._ty, **kw)
        self._withins = []
        self._efficiencies = []
        for scint in scints:
            w, e = scint.within(v, p, **kw)
            self._withins.append(w)
            self._efficiencies.append(e)
        
        self._N = len(self._costheta)
    
    def count(self, *expr):
        expr = list(expr)
        if len(expr) == 0:
            expr = [True]
        if len(expr) == 1:
            expr *= len(self._scints)
        if len(self._scints) != len(expr):
            raise ValueError("expr must have length %d" % len(self._scints))
        if not expr[self._pivot]:
            raise ValueError("the pivot scint %d must be True" % self._pivot)
        expr.pop(self._pivot)
        
        within = np.ones(self._N) * self._pivot_eff
        for i in range(len(self._withins)):
            if expr[i] is None:
                continue
            elif expr[i]:
                w = self._withins[i] * self._efficiencies[i]
            else:
                w = 1 - self._withins[i] * self._efficiencies[i]
            within *= w
        
        count = np.sum(within)
        count_sd = np.std(within, ddof=1) * np.sqrt(len(within))
        
        return un.ufloat(count, count_sd)
    
    def density(self, *expr):
        count = self.count(*expr)
        return self._N / count / self._horizontal_area
    
    def long_run(self, *expr, N=100000000):
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
        
        return N / count / self._horizontal_area

def pmt(idx, efficiency=1.0):
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

    for i in range(len(short_side_offsets)):
        short_side_offsets[i] = un.ufloat(short_side_offsets[i], short_side_offsets_un[i])
    
    i = 6 - idx
    
    return Scint(long_side_length=long_side_lengths[i], short_side_length=short_side_lengths[i], long_side_inclination=long_side_inclinations[i], short_side_inclination=short_side_inclinations[i], short_side_offset=short_side_offsets[i], center_depth=center_depths[i], efficiency=efficiency)

if __name__ == '__main__':
    mc = MC(*[pmt(i) for i in range(6, 0, -1)])
    mc.random_ray()
    mc.run()
    
    print(mc.count())
