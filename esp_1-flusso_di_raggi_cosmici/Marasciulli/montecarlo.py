## MONTE CARLO PER IL FLUSSO DI RAGGI COSMICI
try:
    os.chdir("flusso cosmici")
except FileNotFoundError:
    pass
    
# distribuzione angolare

def distro(teta):
    return  3/(2*pi)*cos(teta)**2 

# cose da definire una volta sola
tutti=10**2
volte=10**4
l1=40 #cm
l2=48 #cm
h=10 #cm
eff=1

print("SIMULAZIONE")
print("Due scintillatori vicini")
print("efficienza=",eff)

# creo un istogramma

py.figure("accettanza geometrica di 2 piani con efficienza 1 con %.0e raggi cosmici ripetuto %.0e volte" %(tutti,volte) ).set_tight_layout(True)
py.rc("font",size=16)
py.grid(color="black",linestyle="--")

py.title("Distribuzione delle accettanze")
py.xlabel("accettanza")
py.ylabel("occorrenza")

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
                
            if c1 and c2 ==True:
                buoni+=1
                
    acc=py.append(acc,buoni/tutti)

py.hist(acc,bins="auto",rwidth=0.9,normed=True)

acce=py.mean(acc) # suppongo che la distribuzione sia gaussiana 
err=py.std(acc,ddof=1) # e lo è se aspetti 3 ore

py.show()

print("accettanza geometrica=",acce*100,"+-",err*100,"%")