import numpy as np
from scipy import optimize, linalg
import numdifftools
from numpy import sin, cos
import lab

class Roba(object):
    def __init__(self, **kwargs):
        for key in kwargs:
            setattr(self, key, kwargs[key])
        

def likelihood_fit(minus_log_likelihood, p0, args=(), **kw):
    result = optimize.minimize(minus_log_likelihood, p0, options=dict(disp=True), args=args, **kw)
    print('computing gradient...')
    gradient = numdifftools.Gradient(minus_log_likelihood)(result.x, *args)
    print('computing hessian...')
    hessian = numdifftools.Hessian(minus_log_likelihood)(result.x, *args)
    
    try:
        cov = linalg.inv(hessian)
    except LinAlgError:
        print('hessian inversion failed, result.hess_inv')
        cov = result.hess_inv
    
    output = Roba(
        par=result.x,
        cov=cov,
        result=result,
        gradient=gradient,
        hessian=hessian
    )
    return output
    
def fun_01(x):
    return 1 / (1 + np.exp(-4 * x))
def fun_10(x):
    return 1 / (1 + np.exp(4 * x))

def signal(xs, mean, L, det_cov, f, fm1, volume):
    res = xs - np.reshape(mean, (3, -1))
    L_res = linalg.solve_triangular(L, res) # inv(L) @ res
    sig = -1/2 * np.log((2 * np.pi) ** 3 * det_cov) - 1/2 * np.einsum('ij,ij->j', L_res, L_res)
    return np.logaddexp(np.log(f) + sig, np.log(fm1 / volume))
    # approssimazione: l'integrale della gaussiana deve fare praticamente 1 nel volume

def minus_log_likelihood(p, xs, volume):
    # extract parameters
    mean = p[:3]
    L_par = p[3:-1]
    f = fun_01(p[-1])
    fm1 = fun_10(p[-1])
    
    # covariance matrix is L @ L.T
    L = np.zeros((3, 3))
    L[np.triu_indices(3)] = L_par
    det_cov = np.prod(np.diag(L)) ** 2
    return -np.sum(signal(xs, mean, L, det_cov, f, fm1, volume))

if __name__ == '__main__':
    from matplotlib import pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    
    fig = plt.figure('likefit')
    fig.clf()
    ax = fig.add_subplot(111, projection='3d')
    
    samples_sig = np.random.randn(3, 1000)
    volume = (np.max(samples_sig) - np.min(samples_sig)) ** 3
    samples_bkg = np.random.uniform(np.min(samples_sig), np.max(samples_sig), size=(3, 100))
    samples = np.concatenate([samples_sig, samples_bkg], axis=1)
    
    ax.scatter(*samples)
    for axis in 'xyz':
        exec('ax.set_{}label("{}")'.format(axis, axis))
    
    p0 = np.array([
        0, 0, 0,
        1, 0, 0, 1, 0, 1,
        0.1
    ])
    output = likelihood_fit(minus_log_likelihood, p0, args=(samples, volume))
    print(lab.format_par_cov(output.par, output.cov))
    
    fig.show()
