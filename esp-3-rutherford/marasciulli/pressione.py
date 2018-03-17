## PRESSIONE

t=10 # secondi
p,c=loadtxt("dati/0316-pressione.txt",unpack=True)

dp=0
dc=sqrt(c)
dc[-1]=1 # non mi piaceva una misura senza errore

figure().set_tight_layout(True)
rc("font",size=14)
grid(linestyle=":")
minorticks_on()

title("Effetto della pressione",size=16)
xlabel("P  [mbar]")
ylabel("conteggi in 10 s")

xscale("log")
errorbar(p,c,xerr=dp,yerr=dc,fmt=" r",marker=".")

show()