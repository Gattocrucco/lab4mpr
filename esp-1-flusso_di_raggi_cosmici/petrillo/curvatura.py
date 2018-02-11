from sympy import symbols, Matrix, sqrt, solveset, simplify

# vedi quaderno data 2018-02-07

c00, c01, c11 = symbols('c00,c01,c11')
C = Matrix([[c00, c01], [c01, c11]])
U = Matrix([[sqrt(c00), sqrt(c11)], [sqrt(c00), -sqrt(c11)]]) # quella sul quaderno è sbagliata!
Ct = U.inv().transpose() @ C @ U.inv()

ctpp = symbols('ctpp')
solution = solveset(Ct[0,0] - ctpp, c01)
print(simplify(solution))
