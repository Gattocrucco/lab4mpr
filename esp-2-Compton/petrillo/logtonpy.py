import numpy as np
import sys

def clear_lines(nlines, nrows):
    for i in range(nlines):
        sys.stdout.write('\033[F\r%s\r' % (" " * nrows,))
    sys.stdout.flush()

def logtonpy(filename, printevery=-1):
    """
    Read .log ADC file <filename> and returns numpy array with type uint16.
    If a line is invalid (not 4 exadecimal characters), an exception is raised.
    """
    values = np.empty(2, dtype='uint16')
    with open(filename, 'r') as logfile:
        i = 0
        for line in logfile:
            if len(line) != 5:
                raise ValueError('line %d (%s) is not 5 characters long' % (i+1, line))
            value = int(line, base=16)
            if i >= len(values):
                values.resize(len(values) * 2)
            values[i] = value
            i += 1
            if printevery > 0 and i % printevery == 0:
                if i > printevery:
                    clear_lines(1, 70)
                print('lines: %d' % i)
    values.resize(i)
    return values

if __name__ == '__main__':
    values = logtonpy(sys.argv[1], printevery=100000)

    npyfilename = filename.replace('.log', '.npy')
    print('saving as uint16 array in file %s' % (npyfilename,))
    np.save(npyfilename, values)
