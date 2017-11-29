from cmath import sin, cos, sqrt, log, tan
from pylab import arctan2, pi

def csc(x):
    return 1/sin(x)
def cot(x):
    return 1/tan(x)

def integ1(x, a):
    return -(csc(x) * sqrt(-2 * a**2 + cos(2 * x) - 1) * log(sqrt(-2 * a**2 + cos(2 * x) - 1) + sqrt(2) * cos(x)))/sqrt(2 * a**2 * csc(x)**2 + 2)

def integ(x, a):
    return 4/pi * (csc(x)**2 * (sqrt(2) * csc(x) * (-2 * a**2 + cos(2 * x) - 1)**(3/2) * log(sqrt(-2 * a**2 + cos(2 * x) - 1) + sqrt(2) * cos(x)) - (2 * a**2 * cot(x) * (-2 * a**2 + cos(2 * x) - 1))/(a**2 + 1)))/(12 * (a**2 * csc(x)**2 + 1)**(3/2))
    
H1 = 20
H2 = 24
h = 20

s12 = integ(arctan2(H2, -H1), h/H2) - integ(arctan2(H2, H1), h/H2)
s21 = integ(arctan2(H1, -H2), h/H1) - integ(arctan2(H1, H2), h/H1)

I = 2 * (s12 + s21)

print(I/(2*pi), ' frazione dei muoni totali')
    