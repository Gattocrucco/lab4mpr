import numpy as np
from scipy import stats
import sympy
import uncertainties as un

def make_sympy_solver():
    Vx = sympy.symbols('Vx:3')
    Vy = sympy.symbols('Vy:3')
    v = sympy.symbols('v:3')
    Pmp = sympy.symbols('Pmp:3')
    S = sympy.Matrix([v, [-vx for vx in Vx], [-vy for vy in Vy], Pmp]).transpose()
    ts = sympy.symbols('t tx ty')
    sol = sympy.solve_linear_system(S, *ts)
    ftx = sympy.lambdify(v + Vx + Vy + Pmp, sol[ts[1]])
    fty = sympy.lambdify(v + Vx + Vy + Pmp, sol[ts[2]])
    return ftx, fty

ftx, fty = make_sympy_solver()

class Scint(object):
    
    def _asufloat(self, x):
        if isinstance(x, un.core.Variable):
            return x
        else:
            return un.ufloat(x, 0)
    
    def __init__(self, long_side_length=480, short_side_length=400, center_depth=0, short_side_offset=0, long_side_inclination=0, short_side_inclination=0):
        
        self._Lx = self._asufloat(short_side_length) / 1000
        self._Ly = self._asufloat(long_side_length) / 1000
        self._z = -self._asufloat(center_depth) / 1000
        self._x = self._asufloat(short_side_offset) / 1000
        self._alpha = self._asufloat(short_side_inclination) * np.pi / 180
        self._beta = self._asufloat(long_side_inclination) * np.pi / 180
            
    def _urandom(self, x):
        return stats.norm.rvs(loc=x.n, scale=x.s)
    
    def within(self, v, p):
        Lx = self._urandom(self._Lx)
        Ly = self._urandom(self._Ly)
        z = self._urandom(self._z)
        x = self._urandom(self._x)
        alpha = self._urandom(self._alpha)
        beta = self._urandom(self._beta)
        
        Vx = np.array([np.cos(alpha), 0, np.sin(alpha)])
        Vy = np.array([np.sin(alpha) * np.sin(beta), np.cos(beta), -np.cos(alpha) * np.sin(beta)])
        P = np.array([x, 0, -z - Lx/2 * np.sin(alpha)])
        
        args = tuple(v.reshape(3,-1))
        args += tuple(Vx.reshape(-1,1))
        args += tuple(Vy.reshape(-1,1))
        args += tuple(P.reshape(-1,1) - p.reshape(3,-1))
        
        tx = ftx(*args)
        ty = fty(*args)
        
        return np.logical_and(np.logical_and(0 <= tx, tx <= Lx), np.logical_and(0 <= ty, ty <= Ly))
    
    def pivot(self, costheta, phi, tx, ty):
        sintheta = np.sqrt(1 - costheta ** 2)
        
        v = np.array([sintheta * np.cos(phi), sintheta * np.sin(phi), costheta])
        
        Lx = self._urandom(self._Lx)
        Ly = self._urandom(self._Ly)
        z = self._urandom(self._z)
        x = self._urandom(self._x)
        alpha = self._urandom(self._alpha)
        beta = self._urandom(self._beta)
        
        Vx = np.array([np.cos(alpha), 0, np.sin(alpha)])
        Vy = np.array([np.sin(alpha) * np.sin(beta), np.cos(beta), -np.cos(alpha) * np.sin(beta)])
        P = np.array([x, 0, -z - Lx/2 * np.sin(alpha)])
        
        p = P.reshape(-1,1) + Vx.reshape(-1,1) * tx.reshape(1,-1) * Lx + Vy.reshape(-1,1) * ty.reshape(1,-1) * Ly
        
        return v, p

class MC(object):
    
    def __init__(self, scints, pivot_scint=0):
        if not isinstance(pivot_scint, Scint):
            self.pivot = scints.pop(pivot_scint)
        else:
            self.pivot = pivot_scint
        self.scints = scints
    
    def random_ray(self, N=1000):
        self._costheta = np.cbrt(stats.uniform.rvs(size=N))
        self._phi = stats.uniform.rvs(size=N, scale=2 * np.pi)
        self._tx = stats.uniform.rvs(size=N)
        self._ty = stats.uniform.rvs(size=N)
            
    def run(self):
        v, p = self.pivot.pivot(self._costheta, self._phi, self._tx, self._ty)
        self.withins = []
        for scint in self.scints:
            self.withins.append(scint.within(v, p))
    
    def count(self, boolexpr=True):
        if isinstance(boolexpr, bool):
            boolexpr = [boolexpr] * len(self.scints)
        within = np.ones(len(self.withins[0]), dtype=bool)
        for i in range(len(self.withins)):
            w = self.withins[i] if boolexpr[i] else np.logical_not(self.withins[i])
            within = np.logical_and(within, w)
        return np.sum(within)

if __name__ == '__main__':
    pmt6 = Scint(long_side_length=un.ufloat(482, 1), short_side_length=un.ufloat(400, 2), long_side_inclination=un.ufloat(0.2, 0.1), short_side_inclination=un.ufloat(0.7, 0.1), center_depth=0, short_side_offset=0)
    
    pmt5 = Scint(long_side_length=un.ufloat(482, 1), short_side_length=un.ufloat(404, 2), long_side_inclination=un.ufloat(0.3, 0.1), short_side_inclination=un.ufloat(1.0, 0.1), center_depth=un.ufloat(102, 2), short_side_offset=un.ufloat(-5, 1))
    
    mc = MC([pmt6, pmt5])
    mc.random_ray()
    mc.run()
    
    print(mc.count())
