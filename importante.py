#caricare ad ogni avvio
from pylab import *
import os
#from numpy.fft import rfft as fft
from math import erf
from subprocess import call
import uncertainties
import uncertainties.umath as u
from uncertainties import ufloat as uf
# import readings
# from readings import graphic  
from scipy.optimize import curve_fit
from uncertainties import unumpy
from uncertainties.unumpy import uarray

os.chdir("C:/Users/Andre/desktop/ANDREA/laboratorio 4/flusso cosmici")

def radn(x,n):
    "radice ennesima"
    return x**(1/n)
    
def gerf(x):
    "error function aggiustata"
    return (erf(x/sqrt(2)))/2
  
from scipy.special import chdtrc #(gradi di libertà,x**2)

def gaussiana(N,x,u,sigma):
    return (N/(sigma*sqrt(2*pi)))*e**(-1*((x-u)**2)/(2*sigma**2))
    
def cov(i,j,mat):
    "Dà la covarianza normalizzata attraverso la matrice 'mat'"
    return mat[i,j]/sqrt(mat[i,i]*mat[j,j])
    

def astd(errori):
    "Calcola l'errore sulla media pesata"
    sotto=0
    
    for i in range(len(errori)):
        sotto+=1/(errori[i])**2
     
    errore_quadro=1/sotto
    
    return sqrt(errore_quadro)
  
    
def errore_resistenze(dati):
    D_R=array([])
    for i in range(len(dati)):
        r=dati[i]
        if r<=200:
            errore=sqrt(0.3**2+((0.8/100)*r)**2)
            D_R=append(D_R,errore)
        if 200<r<=2e3:
            errore=sqrt(1**2+((0.8/100)*r)**2)
            D_R=append(D_R,errore)
        if 2000<r<=20e3:
            errore=sqrt(10**2+((0.8/100)*r)**2)
            D_R=append(D_R,errore)
        if 20e3<r<=200e3:
            errore=sqrt(100**2+((0.8/100)*r)**2)
            D_R=append(D_R,errore)
        if 200e3<r<=2e6:
            errore=sqrt(1000**2+((0.8/100)*r)**2)
            D_R=append(D_R,errore)
        if 2e6<r<=20e6:
            errore=sqrt(20000**2+((1/100)*r)**2)
            D_R=append(D_R,errore)
        if r>20e6:
            errore=sqrt(100000**2+((5/100)*r)**2)
            D_R=append(D_R,errore)
    return D_R
    
def errore_tensioni(array):
    D_V=array([])
    for i in range(len(array)):
        v=array[i]
        if v<=0.2:
            errore=sqrt(100e-6**2+((0.5/100)*v)**2)
            D_V=append(D_V,errore)
        if 0.2<v<=2:
            errore=sqrt(1e-3**2+((0.5/100)*v)**2)
            D_V=append(D_V,errore)
        if 2<v<=20:
            errore=sqrt(10e-3**2+((0.5/100)*v)**2)
            D_V=append(D_V,errore)
        if 20<v<=200:
            errore=sqrt(100e-3**2+((0.5/100)*v)**2)
            D_V=append(D_V,errore)
        if v>200:
            errore=sqrt(2+((0.8/100)*v)**2)
            D_V=append(D_V,errore)
    return D_V

def errore_correnti(vettore):
    D_I=array([])
    for m in range(len(vettore)):
        i=vettore[m]
        if 20e-6<i<=200e-6:
            errore=sqrt(0.1e-6**2+((0.5/100)*i)**2)
            D_I=append(D_I,errore)
        if 200e-6<i<=2e-3:
            errore=sqrt(1e-6**2+((0.5/100)*i)**2)
            D_I=append(D_I,errore)
        if 2e-3<i<=20e-3:
            errore=sqrt(10e-6**2+((0.8/100)*i)**2)
            D_I=append(D_I,errore)
        if 20e-3<i<=0.2:
            errore=sqrt(100e-6**2+((0.8/100)*i)**2)
            D_I=append(D_I,errore)
        if 0.2<i<=2:
            errore=sqrt(1e-3**2+((1.5/100)*i)**2)
            D_I=append(D_I,errore)
        if i>2:
            errore=sqrt(50e-3**2+((2/100)*i)**2)
            D_I=append(D_I,errore)
        if i<=20e-6:
            errore=sqrt(1e-8**2+((2/100)*i)**2)
            D_I=append(D_I,errore)
    return D_I
               
def tensioni_rms(valore):
    D_V=array([])
    for i in range (len(valore)):
        v=valore[i]
        if v<=0.2:
            errore=sqrt(300e-6**2+((1.2/100)*v)**2)
            D_V=append(D_V,errore)
        if 0.2<v<=2:
            errore=sqrt(3e-3**2+((0.8/100)*v)**2)
            D_V=append(D_V,errore)
        if 2<v<=20:
            errore=sqrt(30e-3**2+((0.8/100)*v)**2)
            D_V=append(D_V,errore)
        if 20<v<=200:
            errore=sqrt(300e-3**2+((0.8/100)*v)**2)
            D_V=append(D_V,errore)
        if v>200:
            errore=sqrt(9+((1.2/100)*v)**2)
            D_V=append(D_V,errore)
    return D_V
    
def correnti_rms(scossa):
    D_I=array([])
    for m in range(len(scossa)):
        i=scossa[m]
        if 20e-6<i<=200e-6:
            errore=sqrt(0.3e-6**2+((1.8/100)*i)**2)
            D_I=append(D_I,errore)
        if 200e-6<i<=2e-3:
            errore=sqrt(3e-6**2+((1/100)*i)**2)
            D_I=append(D_I,errore)
        if 2e-3<i<=20e-3:
            errore=sqrt(30e-6**2+((1/100)*i)**2)
            D_I=append(D_I,errore)
        if 20e-3<i<=0.2:
            errore=sqrt(300e-6**2+((1.8/100)*i)**2)
            D_I=append(D_I,errore)
        if 0.2<i<=2:
            errore=sqrt(3e-3**2+((1.8/100)*i)**2)
            D_I=append(D_I,errore)
        if i>2:
            errore=sqrt(70e-3**2+((3/100)*i)**2)
            D_I=append(D_I,errore)
        if i<=20e-6:
            errore=sqrt(70e-6**2+((3/100)*i)**2)
            D_I=append(D_I,errore)
    return D_I
    
'''
def errore_capacità(cond):
    for c in cond:
        dc=sqrt( ((4/100)*c)**2+ 
'''
'''    
# serie di fourier
                                       # onda quadra:bk=0 ck=2/(k*pi) ampiezza=1/2
w=                                     # onda triangolare:bk=4/(pi*k)**2 ck=0 ampiezza=1/2
t=                                     # ricorda lo 0 array   
k=array(range(1,1000,2) 
wk=w*k               
bk=
ck=
a=0
fourier=0
for i in range(0,len(k)):

fourier+= a/2 + bk[i]*cos(wk[i]*t) + ck[i]*sin(wk[i]*t)
'''

def out(x,y,funpam,dy=1,sig=5):
   '''Rimuove gli outliers.
      funpam: funzione con i parametri del fit
      sig: esclude i dati dopo un certo numero di deviazioni standard'''
   global outlier,eletti,cattivi,tanti,pochi
   outlier=abs( (funpam-y)/dy )
   eletti=array([])
   cattivi=array([])
   tanti=array([])
   pochi=array([])
   
   for i in range (len(y)):
      if outlier[i]<sig:
         eletti=append(eletti,y[i])
         tanti=append(tanti,x[i])
      else:
         cattivi=append(cattivi,y[i])
         pochi=append(pochi,x[i])


def med(uarray):
    """prende i valori medi degli uf """
    medi=array([])
    for uf in uarray:
        medi=append(medi,uf.nominal_value)
    return medi
    
def err(uarray):
    """ prende gli errori degli uf """
    errori=array([])
    for uf in uarray:
        errori=append(errori,uf.std_dev)
    return errori
    
def beta(gamma):
    return sqrt( 1-1/gamma**2 )
    
def moda(dati,binni="auto",norma=False):
    """Calcola la moda di un istogramma.
    dati: array non istogrammato
    binni: valore da dare a histogram(bins="auto")
    norma: scegliere se l'istogramma sarà normalizzato
    La funzione restituisce un ufloat che contiene la moda ed il suo errore (semilarghezza del canale)"""
    occ,bor=histogram(dati,bins=binni,normed=norma)
    occ=list(occ)
    indice=occ.index(max(occ))
    moda=(dati[indice+1]+dati[indice])/2
    errore=(bor[indice+1]-bor[indice])/2
    return uf(moda,errore)
    