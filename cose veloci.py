os.chdir("C:/users/Andrea/desktop/andrea/laboratorio 3/esp 11")
t1,V1,t2,V2=py.loadtxt("uscita monostabile.txt",unpack=True)
dt=1e-8

py.figure(1)
py.rc("font",size=20)
py.title("Multivibratore monostabile",fontsize=25)
py.grid(color="black"); py.minorticks_on
py.xlabel("t  ($\mu$s)"); py.ylabel("Tensioni (V) $\pm$3%")

py.errorbar(t1*10**6,V1,xerr=dt*10**6,color="blue",marker=".",linestyle="-",label="V$_{C2}$")
py.errorbar(t2*10**6,V2,xerr=dt*10**6,color="green",marker="+",linestyle="-",label="V$_{OUT-M}$")

py.minorticks_on()
py.legend(fontsize=15)
#py.xscale("log")
py.tight_layout()
py.show()


## scrivere tabelle su latex

h,m=py.loadtxt("Ottica2A_1.txt",unpack=True)

file=open("prova.tex","a")
for j in range(len(a)):
    print( "$ {:.1uL} & {:.1uL} $ ".format(a[j],b[j]),file=file,end="\\\\ \n")
file.close()