import lab4
import numpy as np

class loadadc(object):
    """
    Load and ADC file.
    
    Parameters
    ----------
    filename : string
        Path of file to load.
    
    Members
    -------
    ch1, ch2, ch3 : int array
        Samples.
    tr1, tr2, tr3, c2, c3 : bool array
        Triggers.
    ts : float array
        Timestamps in seconds.
    
    Methods
    -------
    ch(), tr() :
        ch(X) and tr(X) are equivalent to chX and trX.
    """
    def __init__(self, filename):
        ch1, ch2, ch3, tr1, tr2, tr3, c2, c3, ts = lab4.loadtxt(filename, unpack=True, usecols=(0,1,2,4,5,6,8,9,12))
        self.ch1 = np.asarray(ch1, dtype=int)
        self.ch2 = np.asarray(ch2, dtype=int)
        self.ch3 = np.asarray(ch3, dtype=int)
        self.tr1 = tr1 > 500
        self.tr2 = tr2 > 500
        self.tr3 = tr3 > 500
        self.c2 = c2 > 500
        self.c3 = c3 > 500
        self.ts = ts
    
    def ch(self, channel):
        return eval('self.ch{:d}'.format(channel))
    
    def tr(self, channel):
        return eval('self.tr{:d}'.format(channel))
