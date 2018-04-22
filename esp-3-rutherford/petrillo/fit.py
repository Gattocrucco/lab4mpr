import numpy as np
import glob
import lab4
from uncertainties import unumpy as unp
import uncertainties as un

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

def load_spectrum(fileglob, angle):
    """
    Returns
    -------
    edges (missing codes-aware)
    counts (noise-filtered)
    number of noises
    time range [seconds] (including starting or ending noises)
    """
    # obtain filename
    files = glob.glob('../de0_data/{}'.format(fileglob).format('{:g}'.format(angle).replace('-', '_')))
    if len(files) != 1:
        raise RuntimeError('more or less than one file found for {}, angle {}'.format(fileglob, angle))
    file = files[0]
    
    # load data
    rolled_t, ch1, ch2 = np.loadtxt(file, unpack=True)
    t = unroll_time(rolled_t)
    noise_t = find_noise(t)
    noise_ch2 = ch2 != 0
    noise = noise_t | noise_ch2
    
    # count noise
    n_t = np.sum(noise_t)
    n_ch2 = np.sum(noise_ch2)
    n_ol = np.sum(noise_t & noise_ch2)
    
    if n_t + n_ch2 > 0:
        print('        noise: {} stamp, {} ch2, {} both'.format(n_t, n_ch2, n_ol))
    
    # remove noise
    data = ch1[~noise]
    
    # case of empty data
    if len(data) == 0:
        edges = np.array([0, 2 ** 12])
        counts = np.array([0])
    else:
        edges = np.concatenate([[0], 1 + np.arange(2**7) * 2**5, [2**12]])
        counts, _ = np.histogram(data + 0.5, bins=edges)
        assert np.sum(counts) == len(data)

    return edges, counts, np.sum(noise), np.max(t) - np.min(t)

def load_file(filename, spectrglob):
    print(filename.split('/')[-1])
    ang, scaler, clock = np.loadtxt(filename, unpack=True)
    spectra = []
    count = []
    time = []
    for i in range(len(ang)):
        print('    angle {:g}'.format(ang[i]))
        spectra.append(load_spectrum(spectrglob, ang[i]))
        count.append(np.sum(spectra[i][1]))
        time.append(un.ufloat(clock[i], 1) * 1e-3)
        
        # cross checks
        if count[i] + spectra[i][2] != scaler[i]:
            print('        ADC total {:d} != scaler {:d}'.format(int(count[i] + spectra[i][2]), int(scaler[i])))
        if abs(clock[i] - 1000 * spectra[i][3]) / clock[i] > 0.1:
            print('        clock {:.3f} s - ADC range {:.1f} s > 10 %'.format(clock[i] / 1000, spectra[i][3]))
        
        # ad hoc operations
        if filename == '0320-oro5coll1.txt' and ang[i] == 15:
            # nearly complete spectrum with one ch2==0
            count[i] = scaler[i] - 1
        if filename == '0322-oro0.2coll5.txt' and ang[i] == -70:
            # partial spectrum
            count[i] = scaler[i]
        if filename == '0327-all8coll1.txt' and ang[i] == -15:
            # partial spectrum, but I prefer to check noises here
            time[i] = un.ufloat(spectra[i][3], spectra[i][3] / count[i])
        if filename == '0419-oro5coll5.txt' and ang[i] == 40:
            # partial spectrum without noises
            count[i] = scaler[i]
    
    count = np.array(count)
    time = np.array(time)
    rate = unp.uarray(count, np.where(count > 0, np.sqrt(count), 1)) / time
    
    return rate, spectra

for file_data in files:
    load_file('../dati/' + file_data[0], file_data[1])
    