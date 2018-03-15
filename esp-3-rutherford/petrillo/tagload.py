import numpy as np
import scanf

def tagload(filename):
    data = []
    tags = {}
    with open(filename, 'r') as text_file:
        i = 0
        for line in text_file:
            i += 1
            if line.startswith('#'):
                out = scanf.scanf('#%s %f', line)
                if out is None:
                    out = scanf.scanf('# %s %f', line)
                if out is None:
                    print('tagload: warning: comment line %d ignored' % (i,))
                else:
                    tag, value = out
                if not (tag in tags):
                    tags[tag] = []
                tags[tag].append((len(data), value))
            else:
                try:
                    data.append(float(line))
                except ValueError:
                    print('tagload: warning: line %d is not float' % (i,))
    
    data = np.array(data)
    
    unrolled_tags = {}
    for tag in tags:
        l = np.empty(len(data))
        t = np.array(tags[tag])
        idxs = np.concatenate((t[:,0], [len(data)]))
        values = t[:,1]
        for i in range(len(idxs) - 1):
            l[int(idxs[i]):int(idxs[i+1])] = values[i]
        unrolled_tags[tag] = l
    
    return data, unrolled_tags
    