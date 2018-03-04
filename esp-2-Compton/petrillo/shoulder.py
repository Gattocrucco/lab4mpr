import sympy as sp
from matplotlib import pyplot as plt
import numpy as np

def out_photon_energy(E_in, cos_theta):
    return E_in / (1 + E_in/m_e * (1 - cos_theta))

def pdf_cos_theta(E_in, cos_theta):
    P = out_photon_energy(E_in, cos_theta) / E_in
    return 1/2 * P**2 * (P + P**-1 - (1 - cos_theta**2))

m_e, E_0 = sp.symbols('m_e E_0')
cos_theta = sp.symbols('cos(theta)')
E_gamma = sp.symbols('E_gamma')
E_e = sp.symbols('E_e')

dEgamma_dcostheta = out_photon_energy(E_0, cos_theta).diff(cos_theta)

cos_theta_of_E_gamma, = sp.solve(out_photon_energy(E_0, cos_theta) - E_gamma, cos_theta)

pdf_E_gamma = (pdf_cos_theta(E_0, cos_theta) / dEgamma_dcostheta).subs(cos_theta, cos_theta_of_E_gamma)

pdf_E_e = pdf_E_gamma.subs(E_gamma, E_0 - E_e)

# plot
pdf_symb = sp.simplify(pdf_E_e).subs(E_0, 1.33).subs(m_e, 0.511)
pdf = sp.lambdify((E_e,), pdf_symb)
fig = plt.figure('shoulder')
fig.clf()
ax = fig.add_subplot(111)
e = np.linspace(0, 1.33 - float(out_photon_energy(1.33, -1).subs(m_e, 0.511)), 500)
ax.plot(e, pdf(e), '-k')
# fig.show()

# try convolution

def log_gauss(x, s, scale):
    """
    s -> sigma
    scale -> exp(mu)
    """
    return 1 / (s * (x / scale) * sp.sqrt(2 * sp.pi)) * sp.exp(-(sp.log(x / scale) / s)**2 / 2)

def gauss(x, mu, sigma):
    return 1 / (sigma * sp.sqrt(2 * sp.pi)) * sp.exp(-(x-mu)**2 / sigma**2 / 2)

s, scale = sp.symbols('s scale')
mu, sigma = sp.symbols('mu sigma')
x, a, b, c, d = sp.symbols('x a b c d')

# pdf_E_0 = log_gauss(E_0, s, scale)
pdf_E_0 = gauss(E_0, mu, sigma)

# p(E_e|m_e) = ∫ dE_0 p(E_e|E_0,m_e) p(E_0)
integral = sp.Integral((sp.simplify(pdf_E_e) * pdf_E_0).subs(E_0, x), (x, 0, sp.oo))
print(str(integral.subs(mu, a).subs(sigma, b).subs(E_e, c).subs(m_e, d)))
result = integral.doit()
