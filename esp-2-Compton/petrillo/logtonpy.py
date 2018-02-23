import numpy as np
import sys
import scanf

def clear_lines(nlines, nrows):
    for i in range(nlines):
        sys.stdout.write('\033[F\r%s\r' % (" " * nrows,))
    sys.stdout.flush()

filename = sys.argv[1]
printevery = 100000

values = np.empty(2, dtype='uint16')

with open(filename, 'r') as logfile:
    i = 0
    for line in logfile:
        value = int(line, base=16)
        if i >= len(values):
            values.resize(len(values) * 2)
        values[i] = value
        i += 1
        if i % printevery == 0:
            if i > printevery:
                clear_lines(1, 70)
            print('lines: %d' % i)

values.resize(i)
npyfilename = filename.replace('.log', '.npy')
print('saving as uint16 array in file %s' % (npyfilename,))
np.save(npyfilename, values)
