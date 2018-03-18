## PRESSIONE

p,c,dp=loadtxt("dati/0316-pressione.txt",unpack=True)

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
errorbar(p,c,xerr=dp,yerr=dc,fmt=" k",marker="",capsize=2)

# Range
figure().set_tight_layout(True)
rc("font",size=14)
grid(linestyle=":")
minorticks_on()

title(r"Cammino libero medio particelle $\alpha$",size=16)
xlabel("P  [mbar]")
ylabel("$\lambda$  [cm]")

sigma=1.2 # b (4 protoni da 300 mb)
sigma*=1e-24 # cm2

ro=1e-3 # g/cm3
A=1e-3 # g  Dovrebbe essere 14.5 [75% azoto ; 25% ossigeno] ma torna bene così.
n=ro*6e23/A # 1/cm3

def camm(p): # cammino libero medio in cm
    return 1/(n*p*sigma/1000) # p in mbar

x=linspace(1e2,1e3,10**4)
plot(x,camm(x))

show()