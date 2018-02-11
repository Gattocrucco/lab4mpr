import sympy

curv_var = [[sympy.symbols('c%d%d' % (i, j)) for j in range(7)] for i in range(7)]
curv = sympy.Matrix(curv_var)
curv_inv = curv.inv()
