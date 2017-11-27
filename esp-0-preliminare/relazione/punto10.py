from pylab import *

pmts = [2, 4]
vs = [1700, 1800, 1900, 2000]
colors = [3*[0.], 3*[0.7], 3*[0.3], 3*[0.5]]

for i in range(len(pmts)):
    figure('SN pmt %d' % pmts[i], figsize=[ 4.84,  3.49]).set_tight_layout(True)
    clf()
    
    for j in range(len(vs)):
        file = "pmt%d_coinc_%d.txt" % (pmts[i], vs[j])
        data = loadtxt(file)
        
        thrs = data[0]
        S = data[3] + data[4]
        cont = data[1] + data[2]
        N = cont - S
        
        SN = S/N
        dSN = S/N * sqrt(1/S + 1/N)
        
        errorbar(thrs, SN, yerr=dSN, fmt='--.', color=colors[j], label="%d V" % (vs[j],))
    
    yscale('log')
    title('segnale/rumore PMT%d' % (pmts[i],))
    ylabel('S/N')
    xlabel('soglia discr. (mV)')
    legend(fontsize='small')
    grid(linestyle='--')

show()
        