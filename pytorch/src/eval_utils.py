import numpy as np


def root_mean_sq_err(src, tgt):
    '''
    Root mean squared error

    Arg(s):
        src : numpy[float32]
            source array
        tgt : numpy[float32]
            target array
    Returns:
        float : root mean squared error
    '''

    return np.sqrt(np.mean((tgt - src) ** 2))

def mean_abs_err(src, tgt):
    '''
    Mean absolute error

    Arg(s):
        src : numpy[float32]
            source array
        tgt : numpy[float32]
            target array
    Returns:
        float : mean absolute error
    '''

    return np.mean(np.abs(tgt - src))

def inv_root_mean_sq_err(src, tgt):
    '''
    Inverse root mean squared error

    Arg(s):
        src : numpy[float32]
            source array
        tgt : numpy[float32]
            target array
    Returns:
        float : inverse root mean squared error
    '''

    return np.sqrt(np.mean(((1.0 / tgt) - (1.0 / src)) ** 2))

def inv_mean_abs_err(src, tgt):
    '''
    Inverse mean absolute error

    Arg(s):
        src : numpy[float32]
            source array
        tgt : numpy[float32]
            target array
    Returns:
        float : inverse mean absolute error
    '''

    return np.mean(np.abs((1.0 / tgt) - (1.0 / src)))

def mean_abs_rel_err(src, tgt):
    '''
    Mean absolute relative error (normalize absolute error)

    Arg(s):
        src : numpy[float32]
            source array
        tgt : numpy[float32]
            target array
    Returns:
        float : mean absolute relative error between source and target
    '''

    return np.mean(np.abs(src - tgt) / tgt)
