import sympy as sp
from matplotlib import pyplot as plt
import numpy as np

def out_photon_energy(E_in, cos_theta):
    return E_in / (1 + E_in/m_e * (1 - cos_theta))

def pdf_cos_theta(E_in, cos_theta):
    P = out_photon_energy(E_in, cos_theta) / E_in
    return 1/2 * P**2 * (P + P**-1 - (1 - cos_theta**2))

m_e, E_0 = sp.symbols('m_e E_0', positive=True)
cos_theta = sp.symbols('cos(theta)', real=True)
energy = sp.symbols('E_gamma', positive=True)
electron_energy = sp.symbols('E_e', positive=True)

dEdcostheta = out_photon_energy(E_0, cos_theta).diff(cos_theta)

cos_theta_of_energy, = sp.solve(out_photon_energy(E_0, cos_theta) - energy, cos_theta)

pdf_energy = (pdf_cos_theta(E_0, cos_theta) / dEdcostheta).subs(cos_theta, cos_theta_of_energy)

pdf_electron_energy = pdf_energy.subs(energy, E_0 - electron_energy)

pdf_symb = sp.simplify(pdf_electron_energy).subs(E_0, 1.33).subs(m_e, 0.511)

pdf = sp.lambdify((electron_energy,), pdf_symb)

fig = plt.figure('shoulder')
fig.clf()
ax = fig.add_subplot(111)

e = np.linspace(0.511, 1.33 - float(out_photon_energy(1.33, -1).subs(m_e, 0.511)), 500)
ax.plot(e, pdf(e), '-k')

fig.show()
