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
import tensorflow as tf
import tensorflow.contrib.slim as slim


'''
ScaffNet loss functions
'''
def l1_loss_func(src, tgt, v, normalize=False):
    '''
    Computes L1 loss

    Args:
        src : tensor
            N x H x W x D source tensor (prediction)
        tgt : tensor
            N x H x W x D target tensor (groundtruth)
        v : tensor
            N x H x W x 1 map of valid supervision
        normalize : bool
            if set, normalize w.r.t. the target
    Returns:
        float : L1 loss
    '''

    delta = v * tf.abs(src - tgt)

    if normalize:
        delta /= (tgt + 1e-8)

    loss = tf.reduce_sum(delta, axis=[1, 2, 3], keepdims=True)
    loss = loss / tf.reduce_sum(v, axis=[1, 2, 3], keepdims=True)

    return tf.reduce_mean(loss)


'''
FusionNet loss functions
'''
def color_consistency_loss_func(src, tgt, v):
    '''
    Computes color (photometric) consistency loss

    Args:
        src : tensor
            N x H x W x D source image
        tgt : tensor
            N x H x W x D target image
        v : tensor
            N x H x W x 1 map of valid supervision
    Returns:
        float : loss
    '''

    delta = v * tf.abs(src - tgt)
    loss = tf.reduce_sum(delta, axis=[1, 2, 3], keepdims=True)
    loss = loss / tf.reduce_sum(v, axis=[1, 2, 3], keepdims=True)

    return tf.reduce_mean(loss)

def structural_loss_func(src, tgt, v):
    '''
    Computes structural (ssim) consistency loss

    Args:
        src : tensor
            N x H x W x D source image
        tgt : tensor
            N x H x W x D target image
        v : tensor
            N x H x W x 1 map of valid supervision
    Returns:
        float : loss
    '''

    shape = tf.shape(src)[1:3]

    dist = tf.image.resize_nearest_neighbor(ssim(src, tgt), shape)
    loss = tf.reduce_sum(v * dist, axis=[1, 2, 3], keepdims=True)
    loss = loss / tf.reduce_sum(v, axis=[1, 2, 3], keepdims=True)

    return tf.reduce_mean(loss)

def sparse_depth_loss_func(src, tgt, v):
    '''
    Computes sparse depth consistency loss

    Args:
        src : tensor
            N x H x W x 1 source depth
        tgt : tensor
            N x H x W x 1 target sparse depth
        v : tensor
            N x H x W x 1 map of valid supervision
    Returns:
        float : loss
    '''

    delta = v * tf.abs(src - tgt)
    loss = delta / tf.reduce_sum(v)

    return tf.reduce_sum(loss)

def prior_depth_loss_func(src, tgt, w):
    '''
    Computes sparse depth consistency loss

    Args:
        src : tensor
            N x H x W x 1 source depth
        tgt : tensor
            N x H x W x 1 target depth
        v : tensor
            N x H x W x 1 map of valid supervision
    Returns:
        float : loss
    '''

    delta = w * tf.abs(src - tgt)
    return tf.reduce_mean(delta)

def smoothness_loss_func(predict, image):
    '''
    Computes sparse depth consistency loss

    Args:
        predict : tensor
            N x H x W x 1 predictions
        image : N x H x W x D tensor
            N x H x W x D target image
    Returns:
        float : loss
    '''

    predict_gradients_y, predict_gradients_x = gradient_yx(predict)
    image_gradients_y, image_gradients_x = gradient_yx(image)

    weights_x = tf.exp(-tf.reduce_mean(tf.abs(image_gradients_x), 3, keepdims=True))
    weights_y = tf.exp(-tf.reduce_mean(tf.abs(image_gradients_y), 3, keepdims=True))

    smoothness_x = tf.reduce_mean(tf.abs(predict_gradients_x) * weights_x)
    smoothness_y = tf.reduce_mean(tf.abs(predict_gradients_y) * weights_y)

    return smoothness_x + smoothness_y


'''
Utility functions for loss functions
'''
def ssim(A, B):
    '''
    Computes SSIM perceptual metric

    Args:
        A : tensor
            N x H x W x D source image
        B : tensor
            N x H x W x D target image
    Returns:
        float : loss
    '''

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    mu_A = slim.avg_pool2d(A, 3, 1, 'VALID')
    mu_B = slim.avg_pool2d(B, 3, 1, 'VALID')

    sigma_A  = slim.avg_pool2d(A ** 2, 3, 1, 'VALID') - mu_A ** 2
    sigma_B  = slim.avg_pool2d(B ** 2, 3, 1, 'VALID') - mu_B ** 2
    sigma_AB = slim.avg_pool2d(A * B , 3, 1, 'VALID') - mu_A * mu_B

    numer = (2 * mu_A * mu_B + C1) * (2 * sigma_AB + C2)
    denom = (mu_A ** 2 + mu_B ** 2 + C1) * (sigma_A + sigma_B + C2)
    score = numer / denom

    return tf.clip_by_value((1 - score) / 2, 0, 1)

def gradient_yx(T):
    '''
    Takes the image gradient of the tensor

    Args:
        T : tensor
            N x H x W x D image
    Returns:
        float : loss
    '''

    gx = T[:, :, :-1, :] - T[:, :, 1:, :]
    gy = T[:, :-1, :, :] - T[:, 1:, :, :]

    return gy, gx
