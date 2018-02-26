## SIMULAZIONE DEI FOTOPICCHI

# modello

def compton(E,teta):
    m=0.511 #MeV
    return E/(1+E/m*(1-cos(teta)))
    
# variabili geometriche

foto=1.17  # valore del fotopicco in MeV
N=10**5    # numero di fotoni visti
zero=radians(5)   # angolo tra 0 ed il centro del cristallo
delta=radians(2)  #larghezza del cristallo in gradi

# simulazione

# senza errore ADC

a=zero-delta
b=zero+delta
angoli=empty(N)
for i in range(N):
    ang=random.uniform(a,b)  # angoli di scattering accettati dal cristallo
    #ang=random.gauss(zero,delta)   # commenta quella che non vuoi usare
    angoli[i]=ang
    
spettro=compton(foto,angoli)

# aggiunta errore ADC

'''
La funzione che sto usando (guarda sotto) è una
gaussiana (quasi) normalizzata che si trova nel file 
"importante". La uso per modellare la 
risoluzione in energia. 
'''

# grafico

figure(5).set_tight_layout(True)
rc("font",size=14)
title("Simulazione dei fotopicchi a %d° con apertura %.1f°"%(round(degrees(zero)),degrees(delta)),size=16)
grid(color="black",linestyle=":")
minorticks_on()

xlabel("energia  [MeV]")
ylabel("eventi")

occ,marg=histogram(spettro,bins="auto")

marg1=delete(marg,[len(marg)-1])+(marg[2]-marg[1]) # creo un array che moltiplico per la risoluzione

plot(marg1,gaussiana((marg1),1/20,compton(foto,zero),0.02) ) # con 1/20 max(gaussiana)=1

#z=spettro #linspace(0,1.5,1000)
#plot(z,gaussiana(z,1/20,compton(foto,zero),0.02))

xlim(0,1.5)
#ylim(0,max(occ)/10)  # per migliorare la visualizzazione quando ci sonon picchi troppo alti

show()