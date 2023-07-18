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
import numpy as np
import tensorflow as tf
import data_utils
import global_constants as settings


class FusionNetStandaloneDataloader(object):
    '''
    Dataloader class for loading:
    (1) image
    (2) sparse depth

    to run FusionNet standalone without needing to run ScaffNet separately

    Args:
        shape : list[int]
            list of [batch, height, width, channel]
        name : str
            name scope to use
        normalize : bool
            if set, then normalize image between [0, 1]
        n_thread : int
            number of threads to use for fetching data
        prefetch_size : int
            buffer size for prefetching data
    '''

    def __init__(self,
                 shape,
                 name=None,
                 normalize=True,
                 n_thread=settings.N_THREAD,
                 prefetch_size=settings.N_THREAD):

        self.n_batch = shape[0]
        self.n_height = shape[1]
        self.n_width = shape[2]
        self.n_channel = shape[3]
        self.normalize = normalize
        self.n_thread = n_thread
        self.prefetch_size = prefetch_size

        self.scope_name = name if name is not None else 'fusionnet_dataloader'

        with tf.variable_scope(self.scope_name):
            # Set up placeholders for network inputs
            self.load_image_composite_placeholder = tf.placeholder(tf.bool, shape=())

            self.image_placeholder = tf.placeholder(tf.string, shape=[None])
            self.sparse_depth_placeholder = tf.placeholder(tf.string, shape=[None])

            self.dataset = tf.data.Dataset.from_tensor_slices((
                self.image_placeholder,
                self.sparse_depth_placeholder))

            self.dataset = self.dataset \
                .map(self._load_func, num_parallel_calls=self.n_thread) \
                .batch(self.n_batch) \
                .prefetch(buffer_size=self.prefetch_size)

            self.iterator = self.dataset.make_initializable_iterator()
            self.next_element = self.iterator.get_next()

            # Image
            self.next_element[0].set_shape(
                [self.n_batch, self.n_height, self.n_width, self.n_channel])
            # Sparse depth
            self.next_element[1].set_shape(
                [self.n_batch, self.n_height, self.n_width, 2])

    def _load_func(self, image_path, sparse_depth_path):
        '''
        Load function for:
        (1) image
        (2) sparse depth

        Args:
            image_composite_path : str
                path to image composite (triplet)
            sparse_depth_path : str
                path to sparse depth map
        Returns:
            tensor : H x W x 3 RGB image
            tensor : H x W x 2 sparse depth and validity map
        '''

        with tf.variable_scope('load_func'):
            # Load image
            image = tf.cond(
                self.load_image_composite_placeholder,
                lambda : self._load_image_composite_func(image_path)[0],
                lambda : self._load_image_func(image_path))

            # Load sparse depth and validity map
            sparse_depth = self._load_depth_with_validity_map_func(sparse_depth_path)

            return (image, sparse_depth)

    def _load_image_composite_func(self, path):
        '''
        Loads image triplet composite and splits it into three separate images

        Args:
            path : str
                path to depth map
        Returns:
            tensor : H x W x 3 RGB image
            tensor : H x W x 3 RGB image
            tensor : H x W x 3 RGB image
        '''

        with tf.variable_scope('load_image_composite_func'):
            image_composite = tf.to_float(tf.image.decode_png(tf.read_file(path)))

            image1, image0, image2 = tf.split(
                image_composite,
                num_or_size_splits=3,
                axis=1)

            if self.normalize:
                image1 = image1 / 255.0
                image0 = image0 / 255.0
                image2 = image2 / 255.0

            return tf.squeeze(image0), tf.squeeze(image1), tf.squeeze(image2)

    def _load_image_func(self, path):
        '''
        Loads image

        Args:
            path : str
                path to depth map
        Returns:
            tensor : H x W x 3 RGB image
        '''

        with tf.variable_scope('load_image_func'):
            image = tf.to_float(tf.image.decode_png(tf.read_file(path)))

            if self.normalize:
                image = image / 255.0

            return tf.squeeze(image)

    def _load_depth_func(self, path):
        '''
        Loads a depth map

        Args:
            path : str
                path to depth map
        Returns:
            tensor : H x W depth map
        '''

        with tf.variable_scope('load_depth_func'):
            depth = tf.py_func(
                data_utils.load_depth,
                [path],
                [tf.float32])

            return tf.squeeze(depth)

    def _load_validity_map_func(self, path):
        '''
        Loads a validity map

        Args:
            path : str
                path to validity map
        Returns:
            tensor : H x W validity map
        '''

        with tf.variable_scope('load_validity_map_func'):
            validity_map = tf.py_func(
                data_utils.load_validity_map,
                [path],
                [tf.float32])

            return tf.squeeze(validity_map)

    def _load_depth_with_validity_map_func(self, path):
        '''
        Loads a depth map with a validity map

        Args:
            path : str
                path to depth map
        Returns:
            tensor : H x W x 2 depth map with validity map
        '''

        with tf.variable_scope('load_depth_with_validity_map_func'):
            depth, validity_map = tf.py_func(
                data_utils.load_depth_with_validity_map,
                [path],
                [tf.float32, tf.float32])

            return tf.concat([
                tf.expand_dims(depth, axis=-1),
                tf.expand_dims(validity_map, axis=-1)], axis=-1)

    def initialize(self,
                   session,
                   image_paths=None,
                   sparse_depth_paths=None,
                   load_image_composite=True,
                   do_center_crop=False,
                   do_bottom_crop=False):

        assert session is not None

        feed_dict = {
            self.image_placeholder                  : image_paths,
            self.sparse_depth_placeholder           : sparse_depth_paths,
            self.load_image_composite_placeholder   : load_image_composite
        }

        session.run(self.iterator.initializer, feed_dict)


if __name__ == '__main__':
    import os

    # Testing standalone dataloader on KITTI
    image_filepath = \
        os.path.join('validation', 'kitti', 'kitti_val_image.txt')
    sparse_depth_filepath = \
        os.path.join('validation', 'kitti', 'kitti_val_sparse_depth.txt')

    image_paths = data_utils.read_paths(image_filepath)
    sparse_depth_paths = data_utils.read_paths(sparse_depth_filepath)

    n_height = 352
    n_width = 1216

    dataloader = FusionNetStandaloneDataloader(
        name='fusionnet_standalone_dataloader',
        shape=[1, n_height, n_width, 1],
        normalize=True)

    session = tf.Session()

    dataloader.initialize(
        session,
        image_paths=image_paths,
        sparse_depth_paths=sparse_depth_paths,
        load_image_composite=True)

    n_sample = 0
    print('Testing standalone dataloader KITTI using paths from: \n {} \n {}'.format(
        image_filepath,
        sparse_depth_filepath))

    while True:
        try:
            image0, sparse_depth = session.run(dataloader.next_element)

            # Test shapes
            if image0.shape != (1, n_height, n_width, 3):
                print('Path={} Image=0  Shape={}'.format(image_paths[n_sample], image0.shape))
            if sparse_depth.shape != (1, n_height, n_width, 2):
                print('Path={} Shape={}'.format(sparse_depth_paths[n_sample], sparse_depth.shape))

            # Test values
            if np.min(image0) < 0.0:
                print('Path={} Image=0  Min={}'.format(image_paths[n_sample], np.min(image0)))
            if np.max(image0) > 1.0:
                print('Path={} Image=0  Max={}'.format(image_paths[n_sample], np.max(image0)))
            if np.min(sparse_depth[..., 0]) < 0.0:
                print('Path={}  Min={}'.format(sparse_depth_paths[n_sample], np.min(sparse_depth[..., 0])))
            if np.max(sparse_depth[..., 0]) > 256.0:
                print('Path={}  Max={}'.format(sparse_depth_paths[n_sample], np.max(sparse_depth[..., 0])))
            if np.min(sparse_depth[..., 1]) < 0.0:
                print('Path={}  Min={}'.format(sparse_depth_paths[n_sample], np.min(sparse_depth[..., 1])))
            if np.max(sparse_depth[..., 1]) > 256.0:
                print('Path={}  Max={}'.format(sparse_depth_paths[n_sample], np.max(sparse_depth[..., 1])))

            n_sample = n_sample + 1

            print('Processed {} samples...'.format(n_sample), end='\r')

        except tf.errors.OutOfRangeError:
            break

    print('Completed tests for standalone dataloader on KITTI using {} samples'.format(n_sample))
