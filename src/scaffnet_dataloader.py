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


class ScaffNetDataloader(object):
    '''
    Dataloader class for loading:
    (1) sparse depth map with validity map
    (2) dense depth (ground truth)

    Args:
        shape : list[int]
            list of [batch, height, width, channel]
        name : str
            name scope to use
        n_thread : int
            number of threads to use for fetching data
        prefetch_size : int
            buffer size for prefetching data
    '''
    def __init__(self,
                 shape,
                 name=None,
                 is_training=True,
                 n_thread=settings.N_THREAD,
                 prefetch_size=settings.N_THREAD):

        self.n_batch = shape[0]
        self.n_height = shape[1]
        self.n_width = shape[2]
        self.n_channel = shape[3]
        self.is_training = is_training
        self.n_thread = n_thread
        self.prefetch_size = prefetch_size

        self.scope_name = name if name is not None else 'scaffnet_dataloader'

        with tf.variable_scope(self.scope_name):
            # Set up placeholders for entry
            self.sparse_depth_placeholder = tf.placeholder(tf.string, shape=[None])

            if is_training:
                self.ground_truth_placeholder = tf.placeholder(tf.string, shape=[None])

                # Set up crop and data augmentation placeholders
                self.center_crop_placeholder = tf.placeholder(tf.bool, shape=())
                self.bottom_crop_placeholder = tf.placeholder(tf.bool, shape=())
                self.random_horizontal_crop_placeholder = tf.placeholder(tf.bool, shape=())
                self.random_vertical_crop_placeholder = tf.placeholder(tf.bool, shape=())
                self.random_horizontal_flip_placeholder = tf.placeholder(tf.bool, shape=())

            if is_training:
                self.dataset = tf.data.Dataset.from_tensor_slices((
                    self.sparse_depth_placeholder,
                    self.ground_truth_placeholder))

                self.dataset = self.dataset \
                    .map(self._load_func, num_parallel_calls=self.n_thread) \
                    .map(self._crop_func, num_parallel_calls=self.n_thread) \
                    .map(self._horizontal_flip_func, num_parallel_calls=self.n_thread) \
                    .batch(self.n_batch) \
                    .prefetch(buffer_size=self.prefetch_size)
            else:
                self.dataset = tf.data.Dataset.from_tensor_slices((
                    self.sparse_depth_placeholder))

                self.dataset = self.dataset \
                    .map(self._load_func, num_parallel_calls=self.n_thread) \
                    .batch(self.n_batch) \
                    .prefetch(buffer_size=self.prefetch_size)

            self.iterator = self.dataset.make_initializable_iterator()
            self.next_element = self.iterator.get_next()

            if is_training:
                # Sparse depth
                self.next_element[0].set_shape(
                    [self.n_batch, self.n_height, self.n_width, self.n_channel])
                # Ground-truth dense depth
                self.next_element[1].set_shape(
                    [self.n_batch, self.n_height, self.n_width, self.n_channel])
            else:
                # Sparse depth
                self.next_element.set_shape(
                    [self.n_batch, self.n_height, self.n_width, self.n_channel])

    def _load_func(self, sparse_depth_path, ground_truth_path=None):
        '''
        Load function for:
        (1) sparse depth with validity map
        (2) if training, ground truth

        Args:
            sparse_depth_path : str
                path to sparse depth map
            ground_truth_path : str
                path to ground truth
        Returns:
            tensor : H x W x 2 input depth of sparse depth and validity map
            tensor : if training, H x W x 2 ground truth and validity map
        '''

        with tf.variable_scope('load_func'):
            # Load sparse depth and validity map
            input_depth = self._load_depth_with_validity_map_func(sparse_depth_path)

            if self.is_training:
                # Load ground-truth dense depth
                ground_truth = \
                    self._load_depth_with_validity_map_func(ground_truth_path)

                return input_depth, ground_truth
            else:
                return input_depth

    def _crop_func(self, input_depth, ground_truth):
        '''
        Crops input depth and ground truth to specified shape

        Args:
            input_depth : tensor
                H x W x 2 input depth of sparse depth snd validity map
            ground_truth : tensor
                H x W x 2 ground truth of sparse depth snd validity map
        Returns:
            tensor : h x w x 2 input depth of sparse depth snd validity map
            tensor : h x w x 2 ground truth of sparse depth snd validity map
        '''

        def crop_func(in0, in1, random_horizontal_crop, random_vertical_crop):
            shape = tf.shape(in0)

            # Center crop to specified height and width, default bottom centered
            start_height = tf.cond(
                self.center_crop_placeholder,
                lambda: tf.to_int32(tf.to_float(shape[0] - self.n_height) / tf.to_float(2.0)),
                lambda: tf.to_int32(shape[0] - self.n_height))

            start_width = tf.to_float(shape[1] - self.n_width) / tf.to_float(2.0)

            # If we allow augmentation then do random horizontal or vertical shift for crop
            start_height = tf.cond(
                tf.math.logical_and(self.center_crop_placeholder, random_vertical_crop),
                lambda: tf.cast(tf.random_uniform([], 0.0, 2.0 * tf.to_float(start_height)), dtype=tf.int32),
                lambda: tf.to_int32(start_height))

            start_height = tf.cond(
                tf.math.logical_and(self.bottom_crop_placeholder, random_vertical_crop),
                lambda: tf.cast(tf.random_uniform([], 0.0, tf.to_float(start_height)), dtype=tf.int32),
                lambda: tf.to_int32(start_height))

            end_height = self.n_height + start_height

            start_width = tf.cond(
                random_horizontal_crop,
                lambda: tf.cast(tf.random_uniform([], 0.0, 2.0 * start_width), dtype=tf.int32),
                lambda: tf.to_int32(start_width))

            end_width = self.n_width + start_width

            # Apply crop
            in0 = in0[start_height:end_height, start_width:end_width, :]
            in1 = in1[start_height:end_height, start_width:end_width, :]

            return in0, in1

        with tf.variable_scope('crop_func'):
            input_depth, ground_truth = tf.cond(
                tf.math.logical_or(self.center_crop_placeholder, self.bottom_crop_placeholder),
                lambda: crop_func(
                    input_depth,
                    ground_truth,
                    self.random_horizontal_crop_placeholder,
                    self.random_vertical_crop_placeholder),
                lambda: (input_depth, ground_truth))

            return input_depth, ground_truth

    def _horizontal_flip_func(self, input_depth, ground_truth):
        '''
        Performs horizontal flip

        Args:
            input_depth : tensor
                H x W x 2 input depth of sparse depth snd validity map
            ground_truth : tensor
                H x W x 2 ground truth of sparse depth snd validity map
        Returns:
            tensor : H x W x 2 input depth of sparse depth snd validity map
            tensor : H x W x 2 ground truth of sparse depth snd validity map
        '''

        def horizontal_flip_func(in0, in1, do_flip):

            # Perform horizontal flip with 0.50 chance
            in0 = tf.cond(
                do_flip > 0.5,
                lambda: tf.image.flip_left_right(in0),
                lambda: in0)
            in1 = tf.cond(
                do_flip > 0.5,
                lambda: tf.image.flip_left_right(in1),
                lambda: in1)

            return in0, in1

        with tf.variable_scope('horizontal_flip_func'):

            do_flip = tf.random_uniform([], 0.0, 1.0)

            input_depth, ground_truth = tf.cond(
                self.random_horizontal_flip_placeholder,
                lambda: horizontal_flip_func(input_depth, ground_truth, do_flip),
                lambda: (input_depth, ground_truth))

            return input_depth, ground_truth

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
                   sparse_depth_paths=None,
                   ground_truth_paths=None,
                   do_center_crop=False,
                   do_bottom_crop=False,
                   random_horizontal_crop=False,
                   random_vertical_crop=False,
                   random_horizontal_flip=False):

        assert session is not None

        if self.is_training:
            feed_dict = {
                self.sparse_depth_placeholder           : sparse_depth_paths,
                self.ground_truth_placeholder           : ground_truth_paths,
                self.center_crop_placeholder            : do_center_crop,
                self.bottom_crop_placeholder            : do_bottom_crop,
                self.random_horizontal_crop_placeholder : random_horizontal_crop,
                self.random_vertical_crop_placeholder   : random_vertical_crop,
                self.random_horizontal_flip_placeholder : random_horizontal_flip,
            }
        else:
            feed_dict = {
                self.sparse_depth_placeholder           : sparse_depth_paths
            }

        session.run(self.iterator.initializer, feed_dict)


if __name__ == '__main__':
    import os

    # Testing dataloader in training mode on Scenenet
    sparse_depth_filepath = os.path.join(
        'training', 'scenenet', 'scenenet_train_sparse_depth_corner-1.txt')
    ground_truth_filepath = os.path.join(
        'training', 'scenenet', 'scenenet_train_ground_truth_corner-1.txt')

    sparse_depth_paths = data_utils.read_paths(sparse_depth_filepath)
    ground_truth_paths = data_utils.read_paths(ground_truth_filepath)

    n_height = 240
    n_width = 320
    n_point_min = 360

    dataloader = ScaffNetDataloader(
        name='scaffnet_dataloader',
        shape=[1, n_height, n_width, 1],
        is_training=True)

    session = tf.Session()

    dataloader.initialize(
        session,
        sparse_depth_paths=sparse_depth_paths,
        ground_truth_paths=ground_truth_paths,
        do_center_crop=False,
        do_bottom_crop=True,
        random_horizontal_crop=False,
        random_vertical_crop=False,
        random_horizontal_flip=False)

    n_sample = 0
    print('Testing dataloader in training mode using paths from: \n {} \n {}'.format(
        sparse_depth_filepath,
        ground_truth_filepath))

    while True:
        try:
            input_depth, ground_truth = session.run(dataloader.next_element)

            # Test shapes
            if input_depth.shape != (1, n_height, n_width, 2):
                print('Path={}  Shape={}'.format(
                    sparse_depth_paths[n_sample], input_depth.shape))
            if ground_truth.shape != (1, n_height, n_width, 2):
                print('Path={}  Shape={}'.format(
                    ground_truth_paths[n_sample], ground_truth.shape))
            # Test values
            if np.any(np.isnan(input_depth)):
                print('Path={}  contains NaN values'.format(
                    sparse_depth_paths[n_sample]))
            if np.min(input_depth[..., 0]) < 0.0:
                print('Path={}  min value ({})less than 0.0'.format(
                    sparse_depth_paths[n_sample], np.min(input_depth[..., 0])))
            if np.max(input_depth[..., 0]) > 256.0:
                print('Path={}  max value ({}) greater than 256.0'.format(
                    sparse_depth_paths[n_sample], np.max(input_depth[..., 0])))
            if not np.array_equal(np.unique(input_depth[..., 1]), np.array([0, 1])):
                print('Path={}  contains values ({}) outside of [0, 1]'.format(
                    sparse_depth_paths[n_sample], np.unique(input_depth[..., 1])))
            if np.sum(np.where(input_depth[..., 0] > 0, 1, 0)) < n_point_min:
                print('Path={}  contains {} (less than {}) points'.format(
                    sparse_depth_paths[n_sample], np.sum(input_depth[..., 1]), n_point_min))
            if np.sum(input_depth[..., 1]) < n_point_min:
                print('Path={}  contains {} (less than {}) points'.format(
                    sparse_depth_paths[n_sample], np.sum(input_depth[..., 1]), n_point_min))
            if np.any(np.isnan(input_depth)):
                print('Path={}  contains NaN values'.format(
                    sparse_depth_paths[n_sample]))
            if np.any(np.isnan(ground_truth)):
                print('Path={}  contains NaN values'.format(
                    ground_truth_paths[n_sample]))
            if np.min(ground_truth[..., 0]) < 0.0:
                print('Path={}  min value ({}) less than 0.0'.format(
                    ground_truth_paths[n_sample], np.min(ground_truth[..., 0])))
            if np.max(ground_truth[..., 0]) > 256.0:
                print('Path={}  max value ({}) greater than 256.0'.format(
                    ground_truth_paths[n_sample], np.max(ground_truth[..., 0])))
            if np.sum(np.where(ground_truth[..., 0] > 0.0, 1.0, 0.0)) < n_point_min:
                print('Path={}  contains {} (less than {}) points'.format(
                    ground_truth_paths[n_sample], np.sum(ground_truth[..., 1]), n_point_min))
            if np.sum(ground_truth[..., 1]) < n_point_min:
                print('Path={}  contains {} (less than {}) points'.format(
                    ground_truth_paths[n_sample], np.sum(ground_truth[..., 1]), n_point_min))

            n_sample = n_sample + 1

            print('Processed {} samples...'.format(n_sample), end='\r')

        except tf.errors.OutOfRangeError:
            break

    print('Completed tests for dataloader in training mode using {} samples'.format(n_sample))

    # Testing dataloader in inference mode on KITTI
    sparse_depth_filepath = os.path.join(
        'validation', 'kitti', 'kitti_val_sparse_depth.txt')

    sparse_depth_paths = data_utils.read_paths(sparse_depth_filepath)

    n_height = 352
    n_width = 1216

    dataloader = ScaffNetDataloader(
        name='scaffnet_dataloader',
        shape=[1, n_height, n_width, 1],
        is_training=False)

    dataloader.initialize(
        session,
        sparse_depth_paths=sparse_depth_paths)

    n_sample = 0
    print('Testing dataloader in inference mode using paths from: \n {}'.format(
        sparse_depth_filepath))

    while True:
        try:
            input_depth = session.run(dataloader.next_element)

            # Test shapes
            if input_depth.shape != (1, n_height, n_width, 2):
                print('Path={}  Shape={}'.format(
                    sparse_depth_paths[n_sample], input_depth.shape))
            # Test values
            if np.any(np.isnan(input_depth)):
                print('Path={}  contains NaN values'.format(
                    sparse_depth_paths[n_sample]))
            if np.min(input_depth[..., 0]) < 0.0:
                print('Path={}  min value ({})less than 0.0'.format(
                    sparse_depth_paths[n_sample], np.min(input_depth[..., 0])))
            if np.max(input_depth[..., 0]) > 256.0:
                print('Path={}  max value ({}) greater than 256.0'.format(
                    sparse_depth_paths[n_sample], np.max(input_depth[..., 0])))
            if not np.array_equal(np.unique(input_depth[..., 1]), np.array([0, 1])):
                print('Path={}  contains values ({}) outside of [0, 1]'.format(
                    sparse_depth_paths[n_sample], np.unique(input_depth[..., 1])))
            if np.sum(np.where(input_depth[..., 0] > 0, 1, 0)) < n_point_min:
                print('Path={}  contains {} (less than {}) points'.format(
                    sparse_depth_paths[n_sample], np.sum(input_depth[..., 1]), n_point_min))
            if np.sum(input_depth[..., 1]) < n_point_min:
                print('Path={}  contains {} (less than {}) points'.format(
                    sparse_depth_paths[n_sample], np.sum(input_depth[..., 1]), n_point_min))
            if np.any(np.isnan(input_depth)):
                print('Path={}  contains NaN values'.format(
                    sparse_depth_paths[n_sample]))

            n_sample = n_sample + 1

            print('Processed {} samples...'.format(n_sample), end='\r')

        except tf.errors.OutOfRangeError:
            break

    print('Completed tests for dataloader in inference mode using {} samples'.format(n_sample))
