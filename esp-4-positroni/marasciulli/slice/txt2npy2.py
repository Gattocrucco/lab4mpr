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
ch1, ts = lab4.loadtxt(filename, unpack=True, usecols=(0, 12))

print('Saving as {}...'.format(newfilename))
np.savez(newfilename, ch1=np.array(ch1, dtype='uint16'), ts=ts)
