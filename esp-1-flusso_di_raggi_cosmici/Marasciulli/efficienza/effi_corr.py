## EFFICIENZA ESP.1

try:
    os.chdir("efficienza")
except FileNotFoundError:
    pass

sys.stdout=open("risultati_corr.txt","w")

print("EFFICIENZE ESP. 1 \n")
file=(3,4,5)
dt=4e-8
for j in range(len(file)):
    t,r1,r2,r3,c2,c3=py.loadtxt("eff_%d.txt" %file[j],unpack=True)
    T=py.sum(t)/1000
    
    cas2=py.sum( (r1*r3*dt)/T )
    cas3=py.sum( (r1*r2*r3*dt**2)/T )  #coincidenze casuali
    
    eff=py.sum(c3)/py.sum(c2)
    deff=eff*(1-eff)/py.sum(c2)
    
    print("efficienza PMT%d= %f +- %f" %(file[j],eff,deff) )
    print("coincidenze casuali a 2= %.0e" %cas2)
    print("coincidenze casuali a 3= %.0e" %cas3,"\n")
    
sys.stdout.close()
sys.stdout=sys.__stdout__