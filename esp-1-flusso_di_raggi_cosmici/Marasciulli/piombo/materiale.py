## CALCOLI PER MISURE CON PIOMBO
try:
    os.chdir("piombo")
except FileNotFoundError:
    pass


n,t,r5,r4,r3,c2,c3=py.loadtxt("lastre_1.txt",unpack=True)

num=array([0,1,2,4])
l3=uarray([],[])
l2=uarray([],[])
dt=4e-8
for N in num:
    T=py.sum(t[n==N])/1000
    
    C3=uarray( c3[n==N],sqrt(c3[n==N]) ) 
    cont3=py.sum(C3)/T
    l3=py.append(l3,cont3)
    
    C2=uarray( c2[n==N],sqrt(c2[n==N]) ) 
    cont2=py.sum(C2)/T
    l2=py.append(l2,cont2)
    
    if N==0:
        corre=cont3/cont2

py.figure(1).set_tight_layout(True)
py.rc("font",size=16)
py.grid(color="black",linestyle="--")
py.minorticks_on()

py.title("Misura con fogli di piombo",size=18)
py.xlabel("numero di fogli")
py.ylabel("conteggi normalizzati")  # intendo nello stesso intervallo di tempo

l2=l2*corre

# punti coincidenze a 2
py.errorbar(num+0.1,med(l2),yerr=err(l2),color="red",linestyle="",marker=".",capsize=2,label="coincidenze a 2")
# punti coincidenze a 3
py.errorbar(num,med(l3),yerr=err(l3),color="black",linestyle="",marker=".",capsize=2,label="coincidenze a 3")

'''
lim=py.xlim()
z=py.linspace(lim[0],lim[1],500)
# linea gialla
py.errorbar(z,[py.average( (med(l3)),weights=1/err(l3)**2 )]*len(z),linestyle="",color="gray",yerr=astd(err(l3)),ecolor="yellow")
#linea tratteggiata
py.plot(py.xlim(),[py.average( (med(l3)),weights=1/err(l3)**2 )]*2,linestyle="--",color="black")
'''
py.legend(loc="best",fontsize="small")
py.show()