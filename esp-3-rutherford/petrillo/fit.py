import numpy as np
import glob

files = [
    ['0320-oro5coll1.txt', '0320oro5um{}.dat'],
    ['0322-oro0.2coll1.txt', '0322oro0.2um{}.dat'],
    ['0322-oro0.2coll5.txt', '032?oro0.2umcoll5ang{}.dat'],
    ['0327-all8coll1.txt', '0???allcoll1ang{}.dat'],
    ['0412-all8coll5.txt', '041?allcoll5ang{}.dat'],
    ['0419-oro5coll5.txt', '04??oro5umcoll5ang{}.dat']
]

def unroll_time(t):
    if len(t) == 1: 
        return t
    tmax = 6553.5
    # preso da max(t)
    # bisogna sommare 65535 e non 65536 perché min(t) == 0.1
    diff = np.diff(t)
    cycles = np.concatenate([[0], np.cumsum(diff < 0)])
    return t + tmax * cycles

def find_noise(t):
    # t deve essere già "srotolato"
    # restituisce un array di indici che sono rumori

    # controlla che il rate sia abbastanza basso,
    # altrimenti è normale che ci siano eventi con lo stesso timestamp
    rate = len(t) / (np.max(t) - np.min(t))
    if rate > 1/60: # più di uno al minuto
        return np.zeros(len(t), dtype=bool)

    # cerca gruppi di eventi consecutivi con lo stesso timestamp
    dt_zero = np.diff(t) == 0
    noise = np.concatenate([dt_zero, [False]]) | np.concatenate([[False], dt_zero])
    return noise
