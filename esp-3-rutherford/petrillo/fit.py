import numpy as np
import glob
import lab4
import lab
from uncertainties import unumpy as unp
import uncertainties as un
from matplotlib import pyplot as plt
import mc
from scipy import integrate

# non cambiare l'ordine!
files = [
    ['0320-oro5coll1.txt',   '0320oro5um{}.dat'          , '1', 'au', '5', [-10, 15]],
    ['0419-oro5coll5.txt',   '04??oro5umcoll5ang{}.dat'  , '5', 'au', '5', [-10, 15]],
    ['0322-oro0.2coll1.txt', '0322oro0.2um{}.dat'        , '1', 'au', '3', [ -5,  6]],
    ['0322-oro0.2coll5.txt', '032?oro0.2umcoll5ang{}.dat', '5', 'au', '3', [ -5,  6]],
    ['0327-all8coll1.txt',   '0???allcoll1ang{}.dat'     , '1', 'al', '8', [  0,  0]],
    ['0412-all8coll5.txt',   '041?allcoll5ang{}.dat'     , '5', 'al', '8', [  0,  0]]
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

def load_spectrum(fileglob, angle, kde=False):
    """
    Returns
    -------
    edges (missing codes-aware)
    counts (noise-filtered)
    number of noises
    time range [seconds] (including starting or ending noises)
    kde result [optional]
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
    
    if not kde:
        return edges, counts, np.sum(noise), np.max(t) - np.min(t)
    else:
        kde = lab4.credible_interval(data, 0.90)
        return edges, counts, np.sum(noise), np.max(t) - np.min(t), kde

def load_file(filename, spectrglob, **kw):
    print(filename.split('/')[-1])
    ang, scaler, clock = np.loadtxt(filename, unpack=True)
    spectra = []
    count = []
    time = []
    for i in range(len(ang)):
        print('    angle {:g}'.format(ang[i]))
        spectra.append(load_spectrum(spectrglob, ang[i], **kw))
        count.append(np.sum(spectra[i][1]))
        time.append(un.ufloat(clock[i], 1) * 1e-3)
        
        # cross checks
        if count[i] + spectra[i][2] != scaler[i]:
            print('        ADC total {:d} != scaler {:d}'.format(int(count[i] + spectra[i][2]), int(scaler[i])))
        if abs(clock[i] - 1000 * spectra[i][3]) / clock[i] > 0.1:
            print('        clock {:.3f} s - ADC range {:.1f} s > 10 %'.format(clock[i] / 1000, spectra[i][3]))
        
        # ad hoc operations
        if filename.endswith('0320-oro5coll1.txt') and ang[i] == 15:
            # nearly complete spectrum with one ch2==0
            count[i] = scaler[i] - 1
            print('############################', count[i])
        if filename.endswith('0322-oro0.2coll5.txt') and ang[i] == -70:
            # partial spectrum
            count[i] = scaler[i]
            print('############################', count[i])
        if filename.endswith('0327-all8coll1.txt') and ang[i] == -15:
            # partial spectrum, but I prefer to check noises here
            time[i] = un.ufloat(spectra[i][3], spectra[i][3] / count[i])
            print('############################', time[i])
        if filename.endswith('0419-oro5coll5.txt') and ang[i] == 40:
            # partial spectrum without noises
            count[i] = scaler[i]
            print('############################', count[i])
    
    count = np.array(count)
    time = np.array(time)
    rate = unp.uarray(count, np.where(count > 0, np.sqrt(count), 1)) / time
    ang = unp.uarray(ang, 1)
    
    return ang, rate, spectra

l=28.5 # mm
d=31
def f(a, tetax):
    #np.pi/2+np.arctan(a/d)-np.arcsin( (l*np.cos(tetax))/(np.sqrt(a**2+l**2+2*a*l*np.sin(tetax))) )
    return np.arctan(np.tan(tetax)-a/(l*np.cos(tetax))) - np.arctan(a/d)

def fitfun(teta, A, tc):
    "Rutherford con pdf per il collimatore. Richiede variabile globale amax = coll/2"
    teta = np.radians(teta)
    tc = np.radians(tc)
    
    def integrando(a,A,tc,tetax):
        return 1e-5/( np.sin( (f(a,tetax)-tc)/2 ))**4
    
    integrali=np.empty(len(teta))
    for x in range(len(teta)):
        integrali[x]=integrate.quad(integrando,-amax,amax,args=(A,tc,teta[x]))[0]/(2 * amax)
    return A/1e-5 * integrali

fits = []
colors = [[0.6]*3, 'black']
p0s = []

fig = plt.figure('fit', figsize=[9.09, 4.93])
fig.clf()
fig.set_tight_layout(True)

for i in range(len(files)):
    # load data
    file_data = files[i]
    filename = file_data[0]
    spectrglob = file_data[1]
    coll_label = file_data[2]
    nucl_label = file_data[3]
    thick_label = file_data[4]
    excluded_interval = file_data[5]
    ang, rate, spectra = load_file('../dati/' + filename, spectrglob, kde=True)
    
    # # fit
    # anom = unp.nominal_values(ang)
    # cut = (anom <= excluded_interval[0]) | (anom >= excluded_interval[1])
    # amax = float(coll_label) / 2
    # if nucl_label == 'au':
    #     p0 = [1e-3, 1] if coll_label == '1' else [1e-4, 1]
    # else:
    #     p0 = [1e-5, 1] if coll_label == '1' else [1e-5, 1]
    # p0s.append(p0)
    # out = lab.fit_curve(fitfun, unp.nominal_values(ang)[cut], unp.nominal_values(rate)[cut], dx=unp.std_devs(ang)[cut], dy=unp.std_devs(rate)[cut], p0=p0, print_info=1)
    # print('fit success: {}'.format(out.success))
    # print('chisq / dof (p) = {:.1f} / {:d} ({:.3g})'.format(out.chisq, out.chisq_dof, out.chisq_pvalue))
    # print(lab.format_par_cov(out.upar, labels=['ampl', 'center']))
    # fits.append(out)
    
    # create figure
    if i % 2 == 0:
        ax1 = fig.add_subplot(2, 3, 1 + i / 2)
        ax2 = fig.add_subplot(2, 3, 4 + i / 2)
        ax1.set_yscale('log')
    
    # plot data
    label = nucl_label + thick_label + ' coll' + coll_label
    color = colors[i % 2]
    lab4.errorbar(ang, rate, ax=ax1, fmt=',', color=color, label=label)
    # any_excl_new = not all(cut)
    # kw = dict(label='escluso dal fit') if (i % 2 == 0 or (i % 2 == 1 and not any_excl)) else dict()
    # if any_excl_new:
    #     ax1.plot(anom[~cut], unp.nominal_values(rate[~cut]), 'rx', markersize=5, **kw)
    # any_excl = any_excl_new
    
    # # plot fit
    # if i % 2 == 1:
    #     for i in range(2):
    #         out = fits[-2 + i]
    #         color = colors[i]
    #         amax = [0.5, 2.5][i]
    #         space_left = np.linspace(-80, -5, 500) + out.par[1]
    #         space_right = np.linspace(5, 80, 500) + out.par[1]
    #         fitkw = dict(linewidth=0.5, color=color if out.success else 'red', scaley=False, scalex=False)
    #         p = out.par if out.success else p0s[-2 + i]
    #         ax1.plot(space_left, fitfun(space_left, *p), '--', **fitkw)
    #         ax1.plot(space_right, fitfun(space_right, *p), '--', **fitkw)
    #         color = colors[1]
    
    # plot spectra
    mode = np.array([s[4][0] for s in spectra])
    mode_minus = mode - np.array([s[4][1] for s in spectra])
    mode_plus = np.array([s[4][2] for s in spectra]) - mode
    lab4.errorbar(ang, mode, yerr=[mode_minus, mode_plus], fmt=',', capsize=2, color=color, ax=ax2)
    
    # # mc
    # target = eval('mc.target_{}{}'.format(nucl_label, thick_label))
    # coll = eval('mc.coll_{}'.format(coll_label))
    # t, w, e = mc.mc_cached(seed=0, N=1000000, **target, **coll, theta_eps=1)
    # t = np.degrees(t)
    # w /= 100
    # counts, edges, unc_counts = lab4.histogram(t, bins=int(np.sqrt(len(t))), weights=w)
    # ax.errorbar(edges[:-1] + (edges[1] - edges[0]) / 2, counts, yerr=unc_counts, fmt=',', zorder=-1, label='mc {}'.format(label))
    
    # figure decoration
    if i % 2 == 1:
        if i == 1:
            ax1.set_ylabel('rate [s$^{-1}$]')
            ax2.set_ylabel('moda $\\pm$ 90 % [canale ADC]')
        ax2.set_xlabel('angolo [°]')
        ax1.legend(loc='best', fontsize='small')
        ax1.grid(linestyle=':')
        ax2.grid(linestyle=':')

fig.show()
    