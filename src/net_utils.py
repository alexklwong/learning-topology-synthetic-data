import tensorflow as tf


def remove_outliers(sparse_depth, threshold=1.5, ksize=7):
    '''
    Outlier removal by filtering those points with large distance discrepancy

    Args:
        sparse_depth : tensor
            N x H x W x 1 sparse depth map
        threshold : float
            threshold to consider a point an outlier
        ksize : int
            kernel size to use for filtering outliers
    Returns:
        tensor : N x H x W x 1 validity map
    '''

    max_val = tf.reduce_max(sparse_depth) + 100.0

    # We only care about min, so we remove all zeros by setting to max
    sparse_depth_mod = tf.where(
        sparse_depth <= 0.0,
        max_val * tf.ones_like(sparse_depth),
        sparse_depth)

    # Find the neighborhood minimum
    n_pad = int(ksize/2)
    sparse_depth_mod = tf.pad(
        sparse_depth_mod,
        paddings=[[0, 0], [n_pad, n_pad], [n_pad, n_pad], [0, 0]],
        mode='CONSTANT',
        constant_values=max_val)

    patches = tf.extract_image_patches(sparse_depth_mod,
        ksizes=[1, ksize, ksize, 1],
        strides=[1, 1, 1, 1],
        rates=[1, 1, 1, 1],
        padding='VALID')

    sparse_depth_min = tf.reduce_min(patches, axis=-1, keepdims=True)

    # Find mark all possible occlusions as zeros
    return tf.where(
        sparse_depth_min < sparse_depth - threshold,
        tf.zeros_like(sparse_depth),
        sparse_depth)
