## CURVA DI CAVO
try:
    os.chdir("esperienza preliminare")
except FileNotFoundError:
    pass
    
m0=py.mean([6,6,5,7,7,5])
m0=uf(m0,sqrt(m0/6))
m32=py.mean([1,3,1,0,1,1])
m32=uf(m32,sqrt(m32/6))
m40=py.mean([0,0,0,1,0,2])
m40=uf(m40,sqrt(m40/6))
m48=uf(0,0)
m66=uf(0,0)
m16=py.mean([6,1,6,1,8,0])
m16=uf(m16,sqrt(m16/6))

m_32=py.mean([10,1,4,1,1,3])
m_32=uf(m_32,sqrt(m_32/6))
m_40=py.mean([0,0,0,0,0,1])
m_40=uf(m_40,sqrt(m_40/6))
m_16=py.mean([7,4,7,2,3,2])
m_16=uf(m_16,sqrt(m_16/6))
m_48=uf(0,0)

dt=array([0,16,32,40,48,66.5,-32,-40,-48,-16])

cont=uarray([],[])
cont=py.append(cont,m0)
cont=py.append(cont,m16)
cont=py.append(cont,m32)
cont=py.append(cont,m40)
cont=py.append(cont,m48)
cont=py.append(cont,m66)
cont=py.append(cont,m_32)
cont=py.append(cont,m_40)
cont=py.append(cont,m_48)
cont=py.append(cont,m_16)

py.figure(1)
py.rc("font",size=16)
py.grid(linestyle="--",color="black")
py.minorticks_on()

py.title("Curva di cavo",size=18,color="green")
py.xlabel("$\Delta$t  (ns)")
py.ylabel("coincidenze")

py.errorbar(dt,med(cont),err(cont),color="black",marker="o",linestyle="")

py.tight_layout()
py.show()