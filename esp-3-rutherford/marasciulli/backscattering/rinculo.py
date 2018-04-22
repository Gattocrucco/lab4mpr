## BACKSCATTERING
from pylab import *
import uncertainties as unc
from uncertainties import unumpy as unp
from uncertainties import ufloat as uf
from uncertainties.unumpy import nominal_values as nom
from uncertainties.unumpy import std_devs as err
from histofun import find_noise
from histofun import unroll_time

print("\nBACKSCATTERING \n")
# carico i super dati

ang,z,cont,clk=loadtxt('magico.txt',unpack=True,usecols=(0,1,2,3))
file,note=loadtxt('magico.txt',unpack=True,usecols=(4,5),dtype=str)

# formato tipo dati: materiale=[cont,clock/1000,[indici]]
# indici mi dà gli indici degli elementi mergiati dagli array originali

# elementi che mi interessano

tag='doppio_oro'
ind=list(note).index(tag)
oro=[ uf(cont[ind],sqrt(cont[ind])),uf(clk[ind]/1000,0.5),ind ]

tag='dissipatori'
ind=list(note).index(tag)
diss=[ uf(cont[ind],sqrt(cont[ind])),uf(clk[ind]/1000,0.5),ind ]

tag=['doppio_all','doppio_all_2']
ind=[0,0]
ind[0]=list(note).index(tag[0])
ind[1]=list(note).index(tag[1])
all=[ uf(sum(cont[ind]),sum(sqrt(cont[ind]))),uf(sum(clk[ind]/1000),1),ind ]

tag='fondo_backscattering'
ind=list(note).index(tag)
buco=[ uf(cont[ind],sqrt(cont[ind])),uf(clk[ind]/1000,0.5),ind ]

# calcoli backscattering

def evt(var,filearray):
    """Dà gli eventi non rumori (dagli spettri).
    Vuole la struttura dei dati e l'array dove ci sono i file di testo corrispondenti agli indici della struttura.
    Struttura: materiale=[cont,clock/1000,[indici]]"""
    
    spettri=filearray[var[2]]
    
    if isinstance(spettri,str):
        ts,eng=loadtxt('../de0_data/'+spettri,unpack=True,usecols=(0,1))
    else:
        ts=array([])
        eng=array([])
        for rt in spettri:
          ts1,eng1=loadtxt('../de0_data/%s'%rt,unpack=True,usecols=(0,1))
          ts=append(ts,ts1)
          eng=append(eng,eng1)
    
    noise=find_noise(unroll_time(ts))
    return uf(sum(~noise),sqrt(sum(~noise)))

fondo=evt(buco,file)/buco[1]
au=evt(oro,file)/oro[1]
allu=evt(all,file)/all[1]

print("fondo=",fondo)
print("rate oro=",au)
print("rate alluminio=",allu,"\n")

dist_au=au.std_score(fondo)
dist_all=allu.std_score(fondo)

print("tensione oro=",abs(dist_au)," sigma")
print("tensione alluminio=",abs(dist_all)," sigma")