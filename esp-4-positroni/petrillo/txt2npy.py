import numpy as np
import lab4
import sys
import os

filename = sys.argv[1]

if not os.path.exists(filename):
    raise ValueError('File {} does not exist.'.format(filename))

if not filename.endswith('.txt'):
    print('Warning: file {} does not ends in .txt.'.format(filename))
    newfilename = filename + '.npz'
else:
    newfilename = filename[:-len('.txt')] + '.npz'

if os.path.exists(newfilename):
    raise RuntimeError('New file {} already exists.'.format(newfilename))

print('Loading {}...'.format(filename))
ch1, ch2, ch3, tr1, tr2, tr3, c2, c3, ts = lab4.loadtxt(filename, unpack=True, usecols=(0,1,2,4,5,6,8,9,12))
ch1, ch2, ch3 = np.array([ch1, ch2, ch3], dtype='uint16')
tr1, tr2, tr3, c2, c3 = np.array([tr1 > 500, tr2 > 500, tr3 > 500, c2 > 500, c3 > 500], dtype=bool)

print('Saving as {}...'.format(newfilename))
np.savez(newfilename, ch1=ch1, ch2=ch2, ch3=ch3, tr1=tr1, tr2=tr2, tr3=tr3, c2=c2, c3=c3, ts=ts)
