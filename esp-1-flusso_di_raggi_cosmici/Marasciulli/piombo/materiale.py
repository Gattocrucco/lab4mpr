## CALCOLI PER MISURE CON PIOMBO
try:
    os.chdir("piombo")
except FileNotFoundError:
    pass


n,t,tre,quattro,cinque,sei,c3=py.loadtxt("lastre_1.txt",unpack=True)
del tre,quattro,cinque,sei

num=[0,1,2,4]
l=array([])
dl=array([])
for N in num:
    cont=c3[n==N]/t[n==N]*1000
    l=py.append(l, py.average(cont,weights=1/cont) )
    dl=py.append(dl, astd(sqrt(cont)) )
    
py.figure(1).set_tight_layout(True)
py.rc("font",size=16)
py.grid(color="black",linestyle="--")
py.minorticks_off()

py.title("Misura con fogli di piombo",size=18)
py.xlabel("numero di fogli")
py.ylabel("conteggi normalizzati")  # intendo nello stesso intervallo di tempo

py.plot([0,4],[py.average(l,weights=1/dl)]*2,linestyle="--",color="black")
py.errorbar(num,l,yerr=dl,color="black",linestyle="",marker="o",capsize=2)

py.show()