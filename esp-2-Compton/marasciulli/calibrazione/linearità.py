## LINEARITà DELL'ADC
try:
    os.chdir("calibrazione")
except FileNotFoundError:
    pass
    
from scipy.odr import odrpack

scrivi=False
if scrivi==True:
    sys.stdout=open("lin_trig.txt","w")
    
print("LINEARITà CON TRIGGER \n")
    
x,dx,y,dy=loadtxt("estratti_trigger.txt",unpack=True)

# fit odr
def retta(par,X):
    return par[0]*X+par[1]
    
funzione=odrpack.Model(retta)
dati=odrpack.RealData(x,y,sx=dx,sy=dy)
odr=odrpack.ODR(dati,funzione,beta0=(1,0))
out=odr.run()

popt,pcov=out.beta, out.cov_beta
m,q=popt
dm,dq=sqrt(pcov.diagonal())
chi=out.sum_square

print("ang coeff=",m,"+-",dm,"MeV/digit")
print("intercetta=",q,"+-",dq,"MeV")

dof=len(x)-len(popt)
print("chi quadro=",chi,"+-",sqrt(2*dof))
print("p_value=",chdtrc(dof,chi))

# grafico

figure(1).set_tight_layout(True)
rc("font",size=14)
title("Linearità ADC con trigger",size=16)
grid(color="black",linestyle=":")
minorticks_on()

xlabel("energia  [digit]")
ylabel("energia  [MeV]")

errorbar(x,y,xerr=dx,yerr=dy,marker=".",color="black",capsize=2,linestyle="")
z=linspace(2000,8000)
plot(z,retta(popt,z),color="black")

show()

if scrivi==True:
    sys.stdout.close()
    sys.stdout=sys.__stdout__