## PROGRAMMA CHE FA GLI ISTOGRAMMI IN UNITA' FISICHE

cartella=r"C:\Users\andre\Desktop\ANDREA\Laboratorio 4\compton\dati"
nome="22feb-na-trigger"
file=cartella+"/histo-"+nome+".dat"

fis=True
fatt=0.000181

# creazione istogramma

grezzi=loadtxt(file)

nbin=350
massimo=8192
lbin=massimo//nbin+1

dati=zeros(nbin)
for j in range(len(grezzi)):
    indice=j//lbin
    dati[int(indice)]+=grezzi[j]

conv=massimo/len(dati)
X=arange(len(dati))*conv

# istogramma

figure(1).set_tight_layout(True)
rc("font",size=14)
title("Istogramma",size=16)
grid(color="black",linestyle=":")
minorticks_on()

if fis==True:
    xlabel("energia  [MeV]")
else:
    fatt=1
    xlabel("energia  [digit]")

ylabel("conteggi")

bar(X*fatt,dati,width=lbin*fatt)

show()

