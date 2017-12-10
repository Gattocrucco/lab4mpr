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
        
        area = Lx * Ly
        horizontal_area = area * np.cos(alpha) * np.cos(beta)
        
        return v, p, horizontal_area

class MC(object):
    
    def __init__(self, scints):
        self.scints = scints
    
    def random_ray(self, N=1000):
        self._costheta = np.cbrt(stats.uniform.rvs(size=N))
        self._phi = stats.uniform.rvs(size=N, scale=2 * np.pi)
        self._tx = stats.uniform.rvs(size=N)
        self._ty = stats.uniform.rvs(size=N)
            
    def run(self, pivot_scint=0):
        scints = self.scints.copy()
        pivot = scints.pop(pivot_scint)
        self._pivot = pivot_scint
        
        v, p, self._horizontal_area = pivot.pivot(self._costheta, self._phi, self._tx, self._ty)
        self.withins = []
        for scint in scints:
            self.withins.append(scint.within(v, p))
    
    def count(self, *expr):
        expr = list(expr)
        if len(expr) == 0:
            expr = [True]
        if len(expr) == 1:
            expr *= len(self.scints)
        if len(self.scints) != len(expr):
            raise ValueError("expr must have length %d" % len(self.scints))
        if not expr[self._pivot]:
            raise ValueError("the pivot scint %d must be True" % self._pivot)
        expr.pop(self._pivot)
        
        within = np.ones(len(self.withins[0]), dtype=bool)
        for i in range(len(self.withins)):
            if expr[i] is None: continue
            w = self.withins[i] if expr[i] else np.logical_not(self.withins[i])
            within = np.logical_and(within, w)
        return np.sum(within)
    
    def density(self, *expr):
        count = self.count(*expr)
        return len(self._costheta) / count / self._horizontal_area

long_side_angles = np.array([0.2, 0.3, 0.4, 0.3, 0.6, 0.4])
short_side_angles = np.array([0.7, 1.0, 1.2, 0.8, 0.2, 0.8])

long_side_angles -= np.mean(long_side_angles)
short_side_angles -= np.mean(short_side_angles)

long_side_inclination = []
short_side_inclination = []
for beta, alpha in zip(long_side_angles, short_side_angles):
    long_side_inclination.append(un.ufloat(beta, 0.05))
    short_side_inclination.append(un.ufloat(alpha, 0.05))

pmt6 = Scint(long_side_length=un.ufloat(482, 1), short_side_length=un.ufloat(400, 2), long_side_inclination=long_side_inclination[0], short_side_inclination=short_side_inclination[0], center_depth=0, short_side_offset=0)
    
pmt5 = Scint(long_side_length=un.ufloat(482, 1), short_side_length=un.ufloat(404, 2), long_side_inclination=long_side_inclination[1], short_side_inclination=short_side_inclination[1], center_depth=un.ufloat(102, 1), short_side_offset=un.ufloat(-5, 1))
    
pmt4 = Scint(long_side_length=un.ufloat(481, 1), short_side_length=un.ufloat(398, 2), long_side_inclination=long_side_inclination[2], short_side_inclination=short_side_inclination[2], center_depth=un.ufloat(205, 1), short_side_offset=un.ufloat(1, 1))
    
pmt3 = Scint(long_side_length=un.ufloat(480, 1), short_side_length=un.ufloat(400, 2), long_side_inclination=long_side_inclination[3], short_side_inclination=short_side_inclination[3], center_depth=un.ufloat(308, 1), short_side_offset=un.ufloat(1, 1.4))
    
pmt2 = Scint(long_side_length=un.ufloat(481, 1), short_side_length=un.ufloat(395, 2), long_side_inclination=long_side_inclination[4], short_side_inclination=short_side_inclination[4], center_depth=un.ufloat(411, 1), short_side_offset=un.ufloat(2, 1.4))
    
pmt1 = Scint(long_side_length=un.ufloat(480, 1), short_side_length=un.ufloat(398, 2), long_side_inclination=long_side_inclination[5], short_side_inclination=short_side_inclination[5], center_depth=un.ufloat(804, 2), short_side_offset=un.ufloat(0, 2))
    
if __name__ == '__main__':
    mc = MC([pmt6, pmt5, pmt4, pmt3, pmt2, pmt1])
    mc.random_ray()
    mc.run()
    
    print(mc.count())
