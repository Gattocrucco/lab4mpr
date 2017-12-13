from pylab import *

figure('attenuazione')
clf()

letters = ['A', 'B', 'C', 'D']

for number in ['1', '2', '3', '4']:
    m = []
    dm = []
    for letter in letters:
        data = loadtxt("../de0_data/misura_%s%s.dat" % (letter, number), unpack=True)[1]
        m.append(mean(data))
        dm.append(std(data, ddof=1) / sqrt(len(data)))
    errorbar(arange(4), m, yerr=dm, fmt='--', capsize=4, capthick=3, label=number)

title('Attenuazione')
legend(loc=0, fontsize='small')
xlabel('coordinata longitudinale')
xticks(arange(4), letters)
ylabel('media +- std')

show()
