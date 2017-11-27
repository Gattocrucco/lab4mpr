from pylab import *

file3 = ['test_T_pmt3_%d.txt' % i for i in range(3)]
file4 = ['test_T_pmt4_%d.txt' % i for i in range(2)]

labels = ['PMT 3', 'PMT 4']
colors = [3*[0.5], 'black']

figure('test_T', figsize=[ 3.98,  2.6 ]).set_tight_layout(True)
clf()

for j in range(2):
    files = [file3, file4][j]
    
    # load and compute mean
    mu, dmu = empty((2, len(files)))
    for i in range(len(files)):
        data = loadtxt(files[i])
        mu[i] = mean(data)
        dmu[i] = sqrt(mu[i]) / sqrt(len(data))
    
    # plot
    errorbar(1 + arange(len(mu)), mu, yerr=dmu, capsize=2, fmt='.', label=labels[j], color=colors[j])

grid()
xticks([1,2,3])
legend(loc=1, fontsize='small')
xlabel('Presa dati, in ordine cronologico')
ylabel('Media aritmetica dei conteggi')
title('Test accensione')

show()