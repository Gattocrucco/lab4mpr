from pylab import *

# Verifica la formula per convolvere le varie cose.

delta_theta = radians(10)
delta_phi = radians(10)
K = 1
n = 1
d = 1

#####################################################

def rutherford(cos_theta):
    x = cos_theta
    return K / (1 - x) ** 2

def pure_rutherford(theta):
    theta_plus = theta + delta_theta / 2
    theta_minus = theta - delta_theta / 2
    x_plus_minus = [cos(theta_plus), cos(theta_minus)]
    x_plus, x_minus = np.max(x_plus_minus, axis=0), np.min(x_plus_minus, axis=0)
    delta_x = x_plus - x_minus
    return delta_phi * delta_x * rutherford(cos(theta))

def conv_rutherford(theta):
    theta_plus  = theta + delta_theta / 2
    theta_minus = theta - delta_theta / 2
    contains_zero = theta_minus <= 0 <= theta_plus
    if contains_zero:
        P = 1
    else:    
        x_plus_minus = [cos(theta_plus), cos(theta_minus)]
        x_plus, x_minus = max(x_plus_minus), min(x_plus_minus)

        sigma = K * delta_phi * (1/(1 - x_plus) - 1/(1 - x_minus))
        
        P = 1 - exp(-n * d * sigma)
    
    return P

figure('testrf2').set_tight_layout(True)
clf()

theta = linspace(0, 90, 500)
plot(theta, [conv_rutherford(t) for t in radians(theta)], label='conv rf')
plot(theta, pure_rutherford(radians(theta)), label='pure rf', scaley=False)
legend(loc='best')
xlabel('theta [°]')
ylabel('P')

show()
