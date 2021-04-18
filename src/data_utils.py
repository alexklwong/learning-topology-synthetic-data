import os
import numpy as np
from PIL import Image


def log(s, filepath=None, to_console=True):
    '''
    Logs a string to either file or console

    Args:
        s : str
            string to log
        filepath
            output filepath for logging
        to_console : bool
            log to console
    '''

    if to_console:
        print(s)

    if filepath is not None:

        if not os.path.isdir(os.path.dirname(filepath)):
            os.makedirs(os.path.dirname(filepath))

            with open(filepath, 'w+') as o:
               o.write(s + '\n')
        else:
            with open(filepath, 'a+') as o:
                o.write(s + '\n')

def read_paths(filepath):
    '''
    Reads a newline delimited file containing paths

    Args:
        filepath : str
            path to file to be read
    Return:
        list : list of paths
    '''

    path_list = []
    with open(filepath) as f:
        while True:
            path = f.readline().rstrip('\n')

            # If there was nothing to read
            if path == '':
                break

            path_list.append(path)

    return path_list

def write_paths(filepath, paths):
    '''
    Stores line delimited paths into file
    Args:
        filepath : str
            path to file to save paths
        paths : list
            paths to write into file
    '''

    with open(filepath, 'w') as o:
        for idx in range(len(paths)):
            o.write(paths[idx] + '\n')

def load_depth_with_validity_map(path, multiplier=256.0):
    '''
    Loads a depth map from a 16/32-bit PNG file

    Args:
        path : str
            path to 16/32-bit PNG file
        multiplier : float
            depth factor multiplier for saving and loading in 16/32 bit png
    Returns:
        numpy : depth map
        numpy : binary validity map for available depth measurement locations
    '''

    # Loads depth map from 16/32-bit PNG file
    z = np.array(Image.open(path), dtype=np.float32)

    # Assert 16/32-bit (not 8-bit) depth map
    z = z / multiplier
    z[z <= 0] = 0.0
    v = z.astype(np.float32)
    v[z > 0]  = 1.0
    return z, v

def load_depth(path, multiplier=256.0):
    '''
    Loads a depth map from a 16/32-bit PNG file

    Args:
        path : str
            path to 16/32-bit PNG file
        multiplier : float
            depth factor multiplier for saving and loading in 16/32 bit png
    Returns:
        numpy : depth map
    '''

    # Loads depth map from 16/32-bit PNG file
    z = np.array(Image.open(path), dtype=np.float32)

    # Assert 16/32-bit (not 8-bit) depth map
    z = z / multiplier
    z[z <= 0] = 0.0
    return z

def save_depth(z, path, multiplier=256.0):
    '''
    Saves a depth map to a 32-bit PNG file

    Args:
        z : numpy
            depth map
        path : str
            path to store depth map
        multiplier : float
            depth factor multiplier for saving and loading in 16/32 bit png
    '''

    z = np.uint32(z * multiplier)
    z = Image.fromarray(z, mode='I')
    z.save(path)

def load_validity_map(path):
    '''
    Loads a valid map from a 16/32-bit PNG file

    Args:
        path : str
            path to 16/32-bit PNG file
    Returns:
        numpy : binary validity map for available depth measurement locations
    '''

    # Loads depth map from 16/32-bit PNG file
    v = np.array(Image.open(path), dtype=np.float32)
    assert(np.all(np.unique(v) == [0, 256]))
    v[v > 0] = 1

    return v

def save_validity_map(v, path):
    '''
    Saves a valid map to a 32-bit PNG file

    Args:
        v : numpy
            validity map
        path : str
            path to store validity map
    '''

    v[v <= 0] = 0.0
    v[v > 0] = 1.0
    v = np.uint32(v * 256.0)
    v = Image.fromarray(v, mode='I')
    v.save(path)

def load_calibration(path):
    '''
    Loads the calibration matrices for each camera (KITTI) and stores it as map

    Args:
        path : str
            path to file to be read
    Returns:
        dict : map containing camera intrinsics keyed by camera id
    '''
    float_chars = set("0123456789.e+- ")
    data = {}
    with open(path, 'r') as f:
        for line in f.readlines():
            key, value = line.split(':', 1)
            value = value.strip()
            data[key] = value

            if float_chars.issuperset(value):
                try:
                    data[key] = np.asarray([float(x) for x in value.split(' ')])
                except ValueError:
                    pass
    return data

def pad_batch(filepaths, n_batch):
    '''
    Pads the filepaths based on the batch size (n_batch)
    e.g. if n_batch is 8 and number of filepaths is 14, then we pad with 2

    Args:
        filepaths : list
            list of filepaths to be read
        n_batch : int
            number of examples in a batch
    Returns:
        list : list of paths with padding
    '''

    n_samples = len(filepaths)
    if n_samples % n_batch > 0:
        n_pad = n_batch - (n_samples % n_batch)
        filepaths.extend([filepaths[-1]] * n_pad)

    return filepaths

def make_epoch(input_arr, n_batch):
    '''
    Generates a random order and shuffles each list in the input_arr according
    to the order

    Args:
        input_arr : list of lists
            list of lists of inputs
        n_batch : int
            number of examples in a batch
    Returns:
        list : list of lists of shuffled inputs
    '''

    assert len(input_arr)
    n = 0
    for inp in input_arr:
        if inp is not None:
            n = len(inp)
            break

    # At least one of the input arrays is not None
    assert n > 0

    idx = np.arange(n)
    n = (n // n_batch) * n_batch
    np.random.shuffle(idx)
    output_arr = []

    for inp in input_arr:
        output_arr.append(None if inp is None else [inp[i] for i in idx[:n]])

    return output_arr
