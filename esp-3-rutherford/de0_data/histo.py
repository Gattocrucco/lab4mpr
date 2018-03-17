import numpy as np
import matplotlib.pyplot as plt
import sys

filenames = sys.argv[1:]

fig = plt.figure('histo')
fig.clf()
fig.set_tight_layout(True)

if len(filenames) > 1:
	datasets = []
	for filename in filenames:
		print('loading %s...' % (filename,))
		t, ch1, ch2 = np.loadtxt(filename, unpack=True)
		datasets.append(ch1)
	ax = fig.add_subplot(111)
	nbinspow = min(int(np.ceil(np.log2(np.sqrt(max([len(ds) for ds in datasets]))))), 12)
	edges = np.arange(2 ** 12 + 1)[::2 ** (12 - nbinspow)]
	ax.hist(datasets, bins=edges, density=True, histtype='step', label=filenames)
	ax.legend(loc='best', fontsize='small')
	ax.set_xlabel('canale ADC')
	ax.set_ylabel('densita')

elif len(filenames) == 1:
	filename = filenames[0]
	print('loading %s...' % (filename,))
	t, ch1, ch2 = np.loadtxt(filename, unpack=True)
	ax1 = fig.add_subplot(211)
	nbinspow = min(int(np.ceil(np.log2(np.sqrt(len(ch1))))), 12)
	edges = np.arange(2 ** 12 + 1)[::2 ** (12 - nbinspow)]
	ax1.hist(ch1, bins=edges, histtype='step', label=filename)
	ax1.legend(loc='best', fontsize='small')
	ax1.set_ylabel('conteggio')
	ax1.set_xlabel('canale ADC')
	ax2 = fig.add_subplot(212)
	ax2.plot(t, ch1, '.', markersize=2)
	ax2.set_xlabel('tempo')
	ax2.set_ylabel('canale ADC')

else:
	print('no filenames specified.')

fig.show()

