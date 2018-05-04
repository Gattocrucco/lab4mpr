import numpy as np

def histo2d(chx, chy, ax=None, **kw):
    bins = np.arange(0, 1200 // 8) * 8
    H, _, _ = np.histogram2d(chx, chy, bins=[bins, bins])
    
    if not (ax is None):
        ax.imshow(-np.log(1 + H), cmap='gray', extent=[np.min(bins), np.max(bins)] * 2, **kw)
    
    return H, bins

if __name__ == '__main__':
    from matplotlib import pyplot as plt
    import sys
    import lab4
    
    filename = sys.argv[1]
    ch1, ch2, c2 = lab4.loadtxt(filename, usecols=(0, 1, 8), unpack=True)
    
    fig = plt.figure('histo2d')
    fig.clf()
    
    ax = fig.add_subplot(111)
    
    H, bins = histo2d(ch1[c2 > 500], ch2[c2 > 500], ax=ax)
    ax.set_xlabel('ch1 c2')
    ax.set_ylabel('ch2 c2')
    
    fig.show()
    