from pylab import *

figure('attenuazione istogramma')
clf()

letters = ['A', 'B', 'C', 'D']

for number in ['1', '2', '3', '4']:
    datas = []
    for letter in letters:
        datas.append(loadtxt("../de0_data/misura_%s%s.dat" % (letter, number), unpack=True)[1])
    subplot(2, 2, int(number))
    hist(datas, bins='sqrt', normed=True, label=letters)
    legend(loc=0, fontsize='small')
    title(number)

show()
