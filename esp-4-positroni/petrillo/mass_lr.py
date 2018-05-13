import numpy as np
import calibration
import lab4
import lab
import loadadc
from matplotlib import pyplot as plt

def mass_lr(filename, cal_label, cut_beta_ch1, cut_neon_ch1, cut_beta_ch2, cut_neon_ch2, ax=None, **kw):
    data = loadadc.loadadc(filename)

    def gauss(x, peak, mean, sigma):
        return peak * np.exp(-(x - mean)**2 / sigma**2)

    mass = dict()
    for channel in [1, 2]:
        samples = data.ch(channel)[data.tr(channel) & data.c2]
        bins = np.arange(1200 // 8) * 8
        hist, _ = np.histogram(samples, bins)
    
        if not ax is None:
            line, = lab4.bar(bins, hist, ax=ax, label='ch{:d}'.format(channel))
            color = line.get_color()
    
        means = dict()
        for peak in ['beta', 'neon']:
            cut = eval('cut_{:s}_ch{:d}'.format(peak, channel))
            x = (bins[1:] + bins[:-1]) / 2
            y = hist
            dy = np.sqrt(hist)
            sel = (x >= cut[0]) & (x <= cut[1])
        
            p0 = [
                np.max(y[sel]),
                (np.max(x[sel]) + np.min(x[sel])) / 2,
                (np.max(x[sel]) - np.min(x[sel])) / 2
            ]
            tags = dict(beta='stat', neon='cal_corr_hr')[peak]
            out = lab.fit_curve(gauss, x[sel], y[sel], dy=dy[sel], p0=p0, tags=tags, **kw)
            means[peak] = out.upar[1]
        
            if not ax is None:
                xspace = np.linspace(np.min(x[sel]), np.max(x[sel]), 100)
                ax.plot(xspace, gauss(xspace, *out.par), color=color, linestyle=':' if peak == 'beta' else '--', label='fit ch{:d} {:s}'.format(channel, peak), linewidth=2)
    
        mass[channel] = calibration.calibration(means['beta'], 'adc', cal_label, channel)
    
    return mass

if __name__ == '__main__':
    fig = plt.figure('mass_lr')
    fig.clf()
    fig.set_tight_layout(True)
    ax = fig.add_subplot(111)
    
    data = np.atleast_2d(np.loadtxt('../dati/mass_lr.txt', dtype=str))
    for args in data:
        print('_____________{}_____________'.format(args[0]))
        args = ('../DAQ/' + args[0], args[1]) + tuple([eval(a) for a in args[2:]])
        masses = mass_lr(*args, ax=ax, print_info=1)
        for i in [1, 2]:
            print('mass ch{} = {:P}'.format(i, masses[i]))

    ax.legend(loc='best', fontsize='small')
    ax.set_yscale('log')
    fig.show()
