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

## COINCIDENZE A 2

# ciclo di dati
V=array([1700,1800,1900,2000])
files=["pmt4_coinc_1700.txt","pmt4_coinc_1800.txt","pmt4_coinc_1900.txt","pmt4_coinc_2000.txt"]
risu=open("calibr_pmt4.txt","w")
print("CALIBRAZIONE PMT4 \n",file=risu)

py.figure(1)
py.rc("font",size=16)
py.grid(linestyle="--",color="black")
py.minorticks_on()

for k in range(len(files)):
    S1,S2,S3,S4=py.loadtxt(files[k],unpack=True)
    s1=array([S1[0],py.mean((S1[1],S1[2])),py.mean((S1[3],S1[4]))])  # non so fare di meglio
    s2=array([S2[0],py.mean((S2[1],S2[2])),py.mean((S2[3],S2[4]))])
    s3=array([S3[0],py.mean((S3[1],S3[2])),py.mean((S3[3],S3[4]))])
    s4=array([S4[0],py.mean((S4[1],S4[2])),py.mean((S4[3],S4[4]))])
    # riduco l'informazione
    d1=array([s1[0],uf(s1[2],sqrt(s1[2]))/uf(s1[1],sqrt(s1[1]))])
    d2=array([s2[0],uf(s2[2],sqrt(s2[2]))/uf(s2[1],sqrt(s2[1]))])
    d3=array([s3[0],uf(s3[2],sqrt(s3[2]))/uf(s3[1],sqrt(s3[1]))])
    d4=array([s4[0],uf(s4[2],sqrt(s4[2]))/uf(s4[1],sqrt(s4[1]))])
    # raccolgo per fare il grafico
    x=array([d1[0],d2[0],d3[0],d4[0]])
    y=array([d1[1],d2[1],d3[1],d4[1]])
    py.errorbar(x,med(y),err(y),linestyle="--",marker="o",label="%i V"%V[k])
    # li salvo sul file di testo
    print("alimentazione=%i V"%V[k],file=risu)
    for j in range(len(x)):
        print("{:.1f} mV".format(x[j]), "\t rapporto S/(S+N)= {:.1u} ".format(y[j]),file=risu)
    print("\n",file=risu)

py.title("Calibrazione PMT4",size=18)
py.xlabel("soglia discriminatore  (mV)")
py.ylabel("rapporto S/(S+N)")
py.legend(loc="best",fontsize=10)
py.tight_layout()
py.show()
risu.close()