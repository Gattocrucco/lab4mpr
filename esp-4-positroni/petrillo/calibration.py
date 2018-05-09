import numpy as np
import calibration_multiple
import calibration_single
import os
import pickle

def calibration(x, input='adc', label='', channel=0):
    """
    Apply calibration to x.
    
    Parameters
    ----------
    x : array_like
        Array of numbers to be converted. Can be uarray.
    input : string, one of 'adc', 'energy'
        Specify the units of x.
    label : string
        Label identifying the calibration.
    channel : one of 1, 2, 3
        The PMT.
    
    Returns
    -------
    x converted to the other units. This is always an uarray
    and includes uncertainties of the calibration parameters.
    """
    if not channel in [1, 2, 3]:
        raise ValueError('channel ({}) must be 1, 2 or 3.'.format(channel))
    
    cache_file = 'calibration.pickle'
    if not os.path.exists(cache_file):
        with open(cache_file, 'wb') as file:
            pickle.dump(dict(), file)
    if not hasattr(calibration, 'cache'):
        with open(cache_file, 'rb') as file:
            calibration.cache = pickle.load(file)
    cache = calibration.cache

    key = (label, channel)
    if not key in cache:
        mq = calibration_multiple.calibration_multiple(label)[channel - 1]
        cache[key] = mq
        with open(cache_file, 'wb') as file:
            pickle.dump(cache, file)
    else:
        mq = cache[key]
    m, q = mq

    x = np.asarray(x)
    if input == 'adc':
        return (x - q) / m
    elif input == 'energy':
        return m * x + q
    else:
        raise KeyError(input)
    