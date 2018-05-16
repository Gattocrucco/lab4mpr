import numpy as np
import gvar as gv
import lsqfit

y = {                                 # data for the dependent variable
   'data1' : gv.gvar([1.376, 2.010], [[ 0.0047, 0.01], [ 0.01, 0.056]]),
   'data2' : gv.gvar([1.329, 1.582], [[ 0.0047, 0.0067], [0.0067, 0.0136]]),
   'b/a'   : gv.gvar(2.0, 0.5)
   }
x = {                                 # independent variable
   'data1' : np.array([0.1, 1.0]),
   'data2' : np.array([0.1, 0.5])
   }
prior = {}
prior['log(a)'] = gv.log(gv.gvar(0.5, 0.5))
prior['b'] = gv.gvar(0.5, 0.5)

def fcn(x, p):                        # fit function of x and parameters p
  ans = {}
  a = gv.exp(p['log(a)'])
  for k in ['data1', 'data2']:
     ans[k] = gv.exp(a + x[k] * p['b'])
  ans['b/a'] = p['b'] / a
  return ans

# do the fit
fit = lsqfit.nonlinear_fit(data=(x, y), prior=prior, fcn=fcn, debug=True)
print(fit.format(maxline=True))       # print standard summary of fit

p = fit.p                             # best-fit values for parameters
outputs = dict(a=gv.exp(p['log(a)']), b=p['b'])
outputs['b/a'] = p['b']/gv.exp(p['log(a)'])
inputs = dict(y=y, prior=prior)
print(gv.fmt_values(outputs))              # tabulate outputs
print(gv.fmt_errorbudget(outputs, inputs)) # print error budget for outputs
