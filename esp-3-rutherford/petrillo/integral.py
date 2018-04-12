from sympy import *

theta, x = symbols('theta x', real=True)
q = symbols('q', positive=True)
integrand = 1 / (1 - x * cos(theta) - sqrt(1 - x**2) * sin(theta))
integral = integrate(integrand, (x, 1 - q, 1))
