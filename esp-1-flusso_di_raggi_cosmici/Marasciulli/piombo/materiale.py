## CALCOLI PER MISURE CON PIOMBO
try:
    os.chdir("piombo")
except FileNotFoundError:
    pass


n,t,tre,quattro,cinque,sei,c3=py.loadtxt("lastre_1.txt",unpack=True)
del tre,quattro,cinque,sei

num=[0,1,2,4]
l=uarray([],[])
for N in num:
    C3=uarray( c3[n==N],sqrt(c3[n==N]) ) 
    cont=py.sum(C3)/py.sum(t[n==N])*1000
    l=py.append(l,cont)

py.figure(1).set_tight_layout(True)
py.rc("font",size=16)
py.grid(color="black",linestyle="--")
py.minorticks_on()

py.title("Misura con fogli di piombo",size=18)
py.xlabel("numero di fogli")
py.ylabel("conteggi normalizzati")  # intendo nello stesso intervallo di tempo

z=py.linspace(-0.2,4.2,500)
py.errorbar(z,[py.average( (med(l)),weights=1/err(l)**2 )]*len(z),linestyle="",color="gray",yerr=astd(err(l)),ecolor="yellow")
py.plot(py.xlim(),[py.average( (med(l)),weights=1/err(l)**2 )]*2,linestyle="--",color="black")
py.errorbar(num,med(l),yerr=err(l),color="black",linestyle="",marker="o",capsize=2)

py.show()