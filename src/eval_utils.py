from __future__ import with_statement
import numpy as np


def root_mean_sq_err(src, tgt):
    '''
    Root mean squared error

    Args:
        src : numpy
            source array
        tgt : numpy
            target array
    Returns:
        float : root mean squared error
    '''

    return np.sqrt(np.mean((tgt - src) ** 2))

def mean_abs_err(src, tgt):
    '''
    Mean absolute error

    Args:
        src : numpy
            source array
        tgt : numpy
            target array
    Returns:
        float : mean absolute error
    '''
    return np.mean(np.abs(tgt - src))

def inv_root_mean_sq_err(src, tgt):
    '''
    Inverse root mean squared error

    Args:
        src : numpy
            source array
        tgt : numpy
            target array
    Returns:
        float : inverse root mean squared error
    '''

    return np.sqrt(np.mean((1.0 / tgt - 1.0 / src) ** 2))

def inv_mean_abs_err(src, tgt):
    '''
    Inverse mean absolute error

    Args:
        src : numpy
            source array
        tgt : numpy
            target array
    Returns:
        float : inverse mean absolute error
    '''

    return np.mean(np.abs(1.0 / tgt - 1.0 / src))
