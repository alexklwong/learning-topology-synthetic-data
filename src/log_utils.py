'''
Authors: Alex Wong <alexw@cs.ucla.edu>, Safa Cicek <safacicek@ucla.edu>

If this code is useful to you, please cite the following paper:
A. Wong, S. Cicek, and S. Soatto. Learning topology from synthetic data for unsupervised depth completion.
In the Robotics and Automation Letters (RA-L) 2021 and Proceedings of International Conference on Robotics and Automation (ICRA) 2021

@article{wong2021learning,
    title={Learning topology from synthetic data for unsupervised depth completion},
    author={Wong, Alex and Cicek, Safa and Soatto, Stefano},
    journal={IEEE Robotics and Automation Letters},
    volume={6},
    number={2},
    pages={1495--1502},
    year={2021},
    publisher={IEEE}
}
'''
import os
import tensorflow as tf
import matplotlib.cm


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

def colorize(value, colormap, vmin=None, vmax=None):
    '''
    Maps a grayscale image to a matplotlib colormap

    Args:
        value : tensor
            N x H x W x 1 tensor
        vmin : float
            the minimum value of the range used for normalization
        vmax : float
            the maximum value of the range used for normalization
    Returns:
        tensor : N x H x W x 3 tensor
    '''

    # Normalize
    vmin = tf.reduce_min(value) if vmin is None else vmin
    vmax = tf.reduce_max(value) if vmax is None else vmax
    value = (value - vmin) / (vmax - vmin)  # vmin..vmax

    # Squeeze last dim if it exists
    value = tf.squeeze(value)

    # Quantize
    indices = tf.to_int32(tf.round(value * 255))
    value = tf.gather(colormap, indices)

    return value


def gray2color(gray, colormap, vmin=None, vmax=None):
    '''
    Converts grayscale tensor (image) to RGB colormap

    Args:
        gray : tensor
            N x H x W x 1 grayscale tensor
        colormap : str
            name of matplotlib color map (e.g. plasma, jet, gist_stern)
        vmin : float
            minimum to visualize
        vmax : float
            maximum to visualize
    Returns:
        tensor : N x H x W x 3 color tensor
    '''

    cm = tf.constant(
        matplotlib.cm.get_cmap(colormap).colors,
        dtype=tf.float32)

    if vmin is not None:
        gray = tf.where(gray < vmin, vmin * tf.ones_like(gray), gray)

    if vmax is not None:
        gray = tf.where(gray > vmax, vmax * tf.ones_like(gray), gray)

    return colorize(gray, cm)
