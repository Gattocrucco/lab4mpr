## MONTE CARLO PER IL FLUSSO DI RAGGI COSMICI
try:
    os.chdir("flusso cosmici")
except FileNotFoundError:
    pass

# distribuzione angolare

def distro(teta):
    return  3/(2*pi)*cos(teta)**2 

# cose da definire una volta sola
tutti=10**4
volte=10**3
l1=40   #cm
l2=48   
  
eff=(0.8,0.8,0.8,0.8,0.8)
totale=py.sum(eff)

dist=(10.2,20.5,30.8,41.1,80.4)
h=py.sum(dist)

try:
    file=open("acc_%d_%d.txt"%(len(dist)+1,totale*100),"x")
except FileExistsError:
    a=input("sovrascrivere? s/n \n")
    if a=="n":
        assert 2==0
    else:
      file=open("acc_%d_%d.txt"%(len(dist)+1,totale*100),"w")  
    
sys.stdout=file
    
print("SIMULAZIONE")
print("%d scintillatori" %(len(dist)+1))
print("efficienza=",eff)
print("distanze=",dist,"cm")
print("%.0e raggi cosmici" %tutti)
print("iterato %.0e volte" %volte)
print("")

# simulazione vera

acc=array([])

for i in range(volte):
    
    buoni=0
    i=0 
    
    while (i<tutti):
        teta=random.uniform(-pi/2,pi/2)
        sul=random.uniform(0,1)  # variabile a caso che sceglie se sono dentro o meno

        x=random.uniform(0,l1)
        y=random.uniform(0,l2) # separazione delle variabili (indipendenza da fi)
        
        
    # simulazione
        if sul<=distro(teta):
            i+=1
            
            if -x<h*tan(teta)<l1-x: # accetto su x
                c1=True
            else:
                c1=False
            
            if -y<h*tan(teta)<l2-y: # accetto su y
                c2=True
            else:
                c2=False
                
            fortuna=0
            for j in range(len(eff)):
                fortuna+=random.uniform(0,1)
                
            if (c1 and c2 == True) and (fortuna<=totale):
                buoni+=1
                
    acc=py.append(acc,buoni/tutti)
    
# creo un istogramma
    
py.figure(1).set_tight_layout(True)
py.rc("font",size=16)
py.minorticks_on()

py.title("Distribuzione delle accettanze")
py.xlabel("accettanza")
py.ylabel("occorrenza")

py.hist(acc,bins="auto",rwidth=0.9,normed=True)

acce=py.mean(acc) # suppongo che la distribuzione sia gaussiana 
err=py.std(acc,ddof=1) # e lo è se aspetti 3 ore

py.show()
py.savefig("acc_%d_%d.pdf"%(len(dist)+1,totale*100))

print("accettanza geometrica=",acce*100,"+-",err*100,"%")
file.close()
sys.stdout.close()
sys.stdout=sys.__stdout__
print("accettanza geometrica=",acce*100,"+-",err*100,"%")