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
  
eff=(1,1,1,1,1,1)  # deve avere n elementi
totale=py.sum(eff)

dist=(10.2,20.5,30.8,41.1,80.4)  # deve avere n-1 elementi
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

py.hist(acc,bins="auto",rwidth=0.9)

acce=py.mean(acc) # suppongo che la distribuzione sia gaussiana 
err=py.std(acc,ddof=1) # e lo è se aspetti 3 ore

py.show()
py.savefig("acc_%d_%d.pdf"%(len(dist)+1,totale*100))

print("accettanza geometrica=",acce*100,"+-",err*100,"%")
file.close()
sys.stdout.close()
sys.stdout=sys.__stdout__
print("accettanza geometrica=",acce*100,"+-",err*100,"%")



## SCINTILLATORE PICCOLO SPOSTATO

# cose da definire una volta sola
tutti=10**4
volte=500
l1=40   #cm
l2=48  

r1=10 # distanza dal centro del piano grande in x  (può essere negativa)
r2=20 # stesso in y

l3=2  # lato del miniscint su x
l4=2  # lato del miniscint su y  (sono così barvo da considerare addirittura i rettangoli)
if r1<0: l3=-l3
if r2<0: l4=-l4   # mi serve per scrivere meno formule nel ciclo

eff=(1,1)
totale=py.sum(eff)
h=1   # distanza in verticale

mini_acc=array([])

try:
    file=open("mini_acc_%d.txt"%(totale*100),"x")
except FileExistsError:
    a=input("sovrascrivere? s/n \n")
    if a=="n":
        assert 2==0
    else:
      file=open("mini_acc_%d.txt"%(totale*100),"w")  
    
sys.stdout=file

print("SIMULAZIONE MINISCINT \n")
print("%.0e raggi cosmici" %tutti)
print("iterato %.0e volte" %volte)
print("distanza bordi del miniscint:","rx=",r1,"ry=",r2,"cm")
print("efficienze=",eff)
print("distanza verticale=",h)
print("")


mini_acc=array([])

for i in range(volte):
    
    buoni=0
    i=0 
    
    while (i<tutti):
        teta=random.uniform(-pi/2,pi/2)
        sul=random.uniform(0,1)  # variabile a caso che sceglie se sono dentro o meno

        x=random.uniform(0,l1)
        y=random.uniform(0,l2) # separazione delle variabili (indipendenza da fi)
        
        
    # simulazione
        if sul<=distro(teta):  # la distro te la prendi da sopra (viva pyzo!)
            i+=1
           
            if r1>0:
                if l1/2+r1<x+h*tan(teta)<l1/2+l3+r1:
                   c1=True
                else:
                    c1=False
            else:
                if l1/2+l3+r1<x+h*tan(teta)<l1/2+r1:
                    c1=True
                else:
                    c1=False
            
            if r2>0:
                if l2/2+r2<y+h*tan(teta)<l2/2+l4+r2:
                   c2=True
                else:
                    c2=False
            else:
                if l2/2+l4+r2<y+h*tan(teta)<l2/2+r2:
                    c2=True
                else:
                    c2=False
                    
            
            fortuna=0
            for j in range(len(eff)):
                fortuna+=random.uniform(0,1)
                
            if (c1 and c2 == True) and (fortuna<=totale):
                buoni+=1
                
    mini_acc=py.append(mini_acc,buoni/tutti)
    
# creo un istogramma
    
py.figure(2).set_tight_layout(True)
py.rc("font",size=16)
py.minorticks_on()

py.title("Distribuzione delle accettanze\n con il mini scintillatore")
py.xlabel("accettanza")
py.ylabel("occorrenza")

py.hist(mini_acc,bins="auto",rwidth=0.9)

mini_acce=py.mean(mini_acc) # suppongo che la distribuzione sia gaussiana 
mini_err=py.std(mini_acc,ddof=1) # e lo è se aspetti 3 ore

py.show()
py.savefig("mini_acc_%d.pdf"%(totale*100))

print("accettanza geometrica=",mini_acce*100,"+-",mini_err*100,"%")
file.close()
sys.stdout.close()
sys.stdout=sys.__stdout__
print("accettanza geometrica=",mini_acce*100,"+-",mini_err*100,"%")