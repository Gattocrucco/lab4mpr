import numpy as np
from scipy import stats
import sympy

class Scint(object):
    
    def __init__(self, long_side_length=480, short_side_length=400, center_depth=0, short_side_offset=0, long_side_inclination=0, short_side_inclination=0):
        
        Lx = short_side_length / 1000
        Ly = long_side_length / 1000
        z = -center_depth / 1000
        x = short_side_offset / 1000
        alpha = short_side_inclination * np.pi / 180
        beta = long_side_inclination * np.pi / 180
        
        self.Vx = np.array([np.cos(alpha), 0, np.sin(alpha)])
        self.Vy = np.array([np.sin(alpha) * np.sin(beta), np.cos(beta), -np.cos(alpha) * np.sin(beta)])
        self.P = np.array([x, 0, -z - Lx/2 * np.sin(alpha)])
        self.Lx = Lx
        self.Ly = Ly

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

class MC(object):
    
    def __init__(self, scints, pivot_scint=0):
        if not isinstance(pivot_scint, Scint):
            self.pivot = scints.pop(pivot_scint)
        else:
            self.pivot = pivot_scint
        self.scints = scints
        
    
    def random_ray(self, N=1000):
        costheta = np.cbrt(stats.uniform.rvs(size=N))
        phi = stats.uniform.rvs(size=N, scale=2 * np.pi)
        sintheta = np.sqrt(1 - costheta ** 2)
        self.v = np.array([sintheta * np.cos(phi), sintheta * np.sin(phi), costheta])
        
        tx = stats.uniform.rvs(size=N, scale=self.pivot.Lx)
        ty = stats.uniform.rvs(size=N, scale=self.pivot.Ly)
        self.p = self.pivot.P.reshape(-1,1) + self.pivot.Vx.reshape(-1,1) * tx.reshape(1,-1) + self.pivot.Vy.reshape(-1,1) * ty.reshape(1,-1)
            
    def run(self):
        self.withins = []
        for scint in self.scints:
            args = tuple(self.v)
            args += tuple(scint.Vx.reshape(-1,1))
            args += tuple(scint.Vy.reshape(-1,1))
            args += tuple(scint.P.reshape(-1,1) - self.p)
            tx = ftx(*args)
            ty = fty(*args)
            within = np.logical_and(np.logical_and(0 <= tx, tx <= scint.Lx), np.logical_and(0 <= ty, ty <= scint.Ly))
            self.withins.append(within)
    
    def count(self, boolexpr=True):
        if isinstance(boolexpr, bool):
            boolexpr = [boolexpr] * len(self.scints)
        within = np.ones(len(self.withins[0]), dtype=bool)
        for i in range(len(self.withins)):
            w = self.withins[i] if boolexpr[i] else np.logical_not(self.withins[i])
            within = np.logical_and(within, w)
        return np.sum(within)

if __name__ == '__main__':
    scintA = Scint(center_depth=0)
    scintB = Scint(center_depth=100)
    mc = MC([scintA, scintB])
    mc.random_ray()
    mc.run()
    print(mc.count())
