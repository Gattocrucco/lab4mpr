## RATE IN TENSIONE
try:
    os.chdir("esperienza preliminare")
except FileNotFoundError:
    pass
    
# dati  (commentare a seconda della situazione)
file=["tensione_PMT2.txt","tensione_PMT3.txt","tensione_PMT4.txt"]
l=3
dati=array([ [1500, 1600, 1700, 1750, 1800, 1850, 1900, 1950,2000],[0 for i in range(9)],   list(range(1600,2100,100)) , [0 for i in range(5)],  list(range(1600,2100,100)) , [0 for i in range(5)]   ] )
for nomi in file:
    if nomi=="tensione_PMT2.txt":
        c1500,c1600,c1700,c1750,c1800,c1850,c1900,c1950,c2000=py.loadtxt(nomi,unpack=True)
        a=[py.mean(c1500),py.mean(c1600),py.mean(c1700),py.mean(c1750),py.mean(c1800),py.mean(c1850),py.mean(c1900),py.mean(c1950),py.mean(c2000)]
        for j in range(9):
            dati[1][j]=a[j]
    else:
        c1600,c1700,c1800,c1900,c2000=py.loadtxt(nomi,unpack=True)
        b=[py.mean(c1600),py.mean(c1700),py.mean(c1800),py.mean(c1900),py.mean(c2000)]
        for j in range(5):
            dati[l][j]=b[j]
        l+=2
    

# grafico
py.figure(1)
py.rc("font",size=16)
py.grid(linestyle="--",color="black")
py.minorticks_on()

py.title("Alimentazione PMT",size=18,color="blue")
py.xlabel("tensione di alimentazione  (V)")
py.ylabel("conteggi")

p=2
for k in range(0,len(dati),2):
    py.errorbar(dati[k],dati[k+1],yerr=sqrt(dati[k+1]),marker="o",label="PMT%i"%p,linestyle="--")
    p+=1

py.tight_layout()
py.legend(loc="upper left")
py.show()
#py.savefig("tensio_pmt.png")

# risultati file di testo
sys.stdout=open("risultati tensione.txt","w")

print("Conteggi in tensione \n")
print("PMT2")
for i in range(len(dati[0])):
    print(dati[0][i],"V","  media=",dati[1][i],"+-",sqrt(dati[1][i]))
print("\nPMT3")
for i in range(len(dati[2])):
    print(dati[2][i],"V","  media=",dati[3][i],"+-",sqrt(dati[3][i]))
print("\nPMT4")
for i in range(len(dati[2])):
    print(dati[2][i],"V","  media=",dati[5][i],"+-",sqrt(dati[5][i]))

sys.stdout.close()
sys.stdout=sys.__stdout__