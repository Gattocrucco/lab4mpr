## EFFICIENZA ESP.1

try:
    os.chdir("efficienza")
except FileNotFoundError:
    pass

#sys.stdout=open("risultati.txt","w")

print("EFFICIENZE ESP. 1 \n")
file=(3,4,5)
for j in range(len(file)):
# ignoro i singoli conteggi
    t,null1,null2,null3,c2,c3=py.loadtxt("eff_%d.txt" %file[j],unpack=True)
    del null1,null2,null3
    
    # misura fatta con il tempo più lungo
    due=c2[-1]
    tre=c3[-1]
    d_due=sqrt(due)
    d_tre=sqrt(tre)
    finale=uf(tre,d_tre)/uf(due,d_due)
    
    # media di tutte le altre (inclusa quella di cui sopra)
    c2=uarray(c2,sqrt(c2))
    c3=uarray(c3,sqrt(c3))
    
    effp=c3/c2
    eff=py.mean(med(effp))
    deff=py.mean(err(effp))
    
    print("efficienza PMT%d="%file[j],eff,"+-",deff)
    print("misura migliore={:.1u}".format(finale))
    print("")
    
#sys.stdout.close()
#sys.stdout=sys.__stdout__