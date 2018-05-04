import numpy as np
import sys
import os

if len(sys.argv) < 2:
    raise RuntimeError('Specify filename as first argument on command line.')

filename = sys.argv[1]

if not os.path.exists(filename):
    raise ValueError('File `{}` does not exist.'.format(filename))

data = np.loadtxt(filename)

filename_noext = filename[:-len('.txt')] if filename.endswith('.txt') else filename
filename_new = filename_noext + '-binary.txt'

if os.path.exists(filename_new):
    raise RuntimeError('File `{}` already exists.'.format(filename_new))

print('Saving as {}...'.format(filename_new))
with open(filename_new, 'w') as file:
    for i in range(data.shape[0]):
        file.write('({:5d}) '.format(i + 1))
        for j in range(data.shape[1] - 1):
            file.write('{:12b}'.format(int(data[i, j])))
        file.write('\n')
