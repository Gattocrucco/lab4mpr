## CONFRONTO

import statistics as stat
# dati
colonna=["A","B","C","D"]
riga=[1,2,3,4]
Y=uarray([],[])

py.figure("prova").set_tight_layout(True)
py.rc("font",size=16)
py.grid(color="black",linestyle=":")
py.minorticks_on()

for j in range(len(colonna)):
    
    mode=array([])
    
    for i in range(len(riga)):

        no,en,non=py.loadtxt("C:/Users/andre/Desktop/ANDREA/Laboratorio 4/flusso cosmici/de0_data/misura_%s%s.dat" %(colonna[j],riga[i]),unpack=True)
        del no,non;
        
        occ,bordi=py.histogram(en,bins="auto")
        occ=list(occ)
        indice=occ.index(max(occ))
        moda=(bordi[indice]+bordi[indice+1])/2
        mode=py.append(mode,moda)
        
        py.hist(en,bins="auto",label="%f"%moda)


    y=py.average(mode)
    dy=astd([3/2**12]*4)   # errore di digitalizzazione in volt
    Y=py.append(Y,uf(y,dy))




'''
py.title("Lunghezza di attenuazione",size=18)
py.xlabel("distanza dalla guida di luce  (cm)")
py.ylabel("valore ADC  (mV)")
 '''   
    
py.legend()
py.show()