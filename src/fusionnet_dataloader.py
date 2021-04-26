import numpy as np
import tensorflow as tf
import data_utils
import global_constants as settings


class FusionNetDataloader(object):
    '''
    Dataloader class for loading:
    (1) image at time t
    (2) image at time t - 1
    (3) image at time t + 1
    (4) input (predicted) depth with sparse depth at time t
    (5) 3 x 3 intrinsics matrix

    Args:
        shape : list[int]
            list of [batch, height, width, channel]
        name : str
            name scope to use
        sparse_input_type : str
            whether to load sparse depth or validity map in place of sparse depth
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
            self.image_composite_placeholder = tf.placeholder(tf.string, shape=[None])
            self.input_depth_placeholder = tf.placeholder(tf.string, shape=[None])
            self.sparse_depth_placeholder = tf.placeholder(tf.string, shape=[None])
            self.intrinsics_placeholder = tf.placeholder(tf.string, shape=[None])

            # Set up placeholder for loading depth
            self.depth_load_multiplier_placeholder = tf.placeholder(tf.float32, shape=())

            # Set up crop and data augmentation placeholders
            self.center_crop_placeholder = tf.placeholder(tf.bool, shape=())
            self.bottom_crop_placeholder = tf.placeholder(tf.bool, shape=())
            self.random_horizontal_crop_placeholder = tf.placeholder(tf.bool, shape=())
            self.random_vertical_crop_placeholder = tf.placeholder(tf.bool, shape=())

            self.dataset = tf.data.Dataset.from_tensor_slices((
                self.image_composite_placeholder,
                self.input_depth_placeholder,
                self.sparse_depth_placeholder,
                self.intrinsics_placeholder))

            self.dataset = self.dataset \
                .map(self._load_func, num_parallel_calls=self.n_thread) \
                .map(self._crop_func, num_parallel_calls=self.n_thread) \
                .batch(self.n_batch) \
                .prefetch(buffer_size=self.prefetch_size)
            self.iterator = self.dataset.make_initializable_iterator()
            self.next_element = self.iterator.get_next()

            # Image 0 (t)
            self.next_element[0].set_shape(
                [self.n_batch, self.n_height, self.n_width, self.n_channel])
            # Image 1 (t - 1)
            self.next_element[1].set_shape(
                [self.n_batch, self.n_height, self.n_width, self.n_channel])
            # Image 2 (t + 1)
            self.next_element[2].set_shape(
                [self.n_batch, self.n_height, self.n_width, self.n_channel])
            # Input depth and sparse depth
            self.next_element[3].set_shape(
                [self.n_batch, self.n_height, self.n_width, 2])
            # Camera intrinsics 3x3 matrix
            self.next_element[4].set_shape([self.n_batch, 3, 3])

    def _load_func(self,
                   image_composite_path,
                   input_depth_path,
                   sparse_depth_path,
                   intrinsics_path):
        '''
        Load function for:
        (1) image at time t
        (2) image at time t - 1
        (3) image at time t + 1
        (4) input depth at time t
        (5) 3 x 3 intrinsics matrix

        Args:
            image_composite_path : str
                path to image composite (triplet)
            input_depth_path : str
                path to dense depth map
            sparse_depth_path : str
                path to sparse depth map
            intrinsics_path : str
                path to 3 x 3 camera intrinsics
        Returns:
            tensor : H x W x 3 RGB image at time t
            tensor : H x W x 3 RGB image at time t - 1
            tensor : H x W x 3 RGB image at time t + 1
            tensor : H x W x 2 input (predicted) depth and sparse depth/validity map
            tensor : 3 x 3 camera intrinsics matrix
            tensor : H x W x 2 ground truth and validity map
        '''

        with tf.variable_scope('load_func'):
            # Load image at time 0, 1, 2
            image0, image1, image2 = \
                self._load_image_composite_func(image_composite_path)

            # Load input (predicted) depth and sparse depth
            input_depth = self._load_depth_func(input_depth_path)

            sparse_depth = self._load_depth_func(sparse_depth_path)

            # Depth and sparse depth or validity map pair
            input_depth = tf.concat([
                tf.expand_dims(input_depth, axis=-1),
                tf.expand_dims(sparse_depth, axis=-1)], axis=-1)

            # Load camera intrinsics
            intrinsics = self._load_intrinsics_func(intrinsics_path)

            return (image0,
                    image1,
                    image2,
                    input_depth,
                    intrinsics)

    def _crop_func(self,
                   image0,
                   image1,
                   image2,
                   input_depth,
                   intrinsics):
        '''
        Crops images, input depth and ground truth to specified shape

        Args:
            image0 : tensor
                H x W x 3 RGB image at time t
            image1 : tensor
                H x W x 3 RGB image at time t - 1
            image2 : tensor
                H x W x 3 RGB image at time t + 1
            input_depth : tensor
                H x W x 2 input depth of sparse depth and validity map
            intrinsics : tensor
                3 x 3 camera intrinsics matrix
        Returns:
            tensor : h x w x 3 RGB image at time t
            tensor : h x w x 3 RGB image at time t - 1
            tensor : h x w x 3 RGB image at time t + 1
            tensor : h x w x 2 input depth of dense depth and sparse depth/validity map
            tensor : 3 x 3 camera intrinsics matrix
        '''

        def crop_func(in0, in1, in2, in3, k, random_horizontal_crop, random_vertical_crop):
            # Center crop to specified height and width, default bottom centered
            shape = tf.shape(in0)

            start_height = tf.cond(
                self.center_crop_placeholder,
                lambda: tf.to_int32(tf.to_float(shape[0] - self.n_height) / tf.to_float(2.0)),
                lambda: tf.to_int32(shape[0] - self.n_height))
            end_height = self.n_height + start_height

            start_width = tf.to_float(shape[1] - self.n_width) / tf.to_float(2.0)

            # If we allow augmentation then do random horizontal or vertical shift for crop
            start_height = tf.cond(
                tf.math.logical_and(self.center_crop_placeholder, random_vertical_crop),
                lambda: tf.cast(tf.random_uniform([], 0.0, 2.0 * start_height), dtype=tf.int32),
                lambda: tf.to_int32(start_height))

            start_height = tf.cond(
                tf.math.logical_and(self.bottom_crop_placeholder, random_vertical_crop),
                lambda: tf.cast(tf.random_uniform([], 0.0, start_height), dtype=tf.int32),
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
            in2 = in2[start_height:end_height, start_width:end_width, :]
            in3 = in3[start_height:end_height, start_width:end_width, :]

            # Adjust camera intrinsics after crop
            k_adj = tf.to_float([
                [0, 0, -start_width ],
                [0, 0, -start_height],
                [0, 0, 0            ]])
            k = k + k_adj

            return in0, in1, in2, in3, k

        with tf.variable_scope('crop_func'):
            image0, image1, image2, input_depth, intrinsics, = tf.cond(
                tf.math.logical_or(self.center_crop_placeholder, self.bottom_crop_placeholder),
                lambda: crop_func(
                    image0,
                    image1,
                    image2,
                    input_depth,
                    intrinsics,
                    self.random_horizontal_crop_placeholder,
                    self.random_vertical_crop_placeholder),
                lambda: (image0, image1, image2, input_depth, intrinsics))

            return (image0,
                    image1,
                    image2,
                    input_depth,
                    intrinsics)

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
                [path, self.depth_load_multiplier_placeholder],
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
                [path, self.depth_load_multiplier_placeholder],
                [tf.float32, tf.float32])

            return tf.concat([
                tf.expand_dims(depth, axis=-1),
                tf.expand_dims(validity_map, axis=-1)], axis=-1)

    def _load_intrinsics_func(self, path):
        '''
        Loads camera intrinsics

        Args:
            path : str
                path to intrinsics
        Returns:
            tensor : 3 x 3 camera intrinsics matrix
        '''

        with tf.variable_scope('load_intrinsics_func'):
            return tf.reshape(self._load_npy_func(path), [3, 3])

    def _load_npy_func(self, path):
        '''
        Loads a numpy file

        Args:
            path : str
                path to numpy (npy)
        Returns:
            tensor : numpy file
        '''
        with tf.variable_scope('load_npy_func'):
            data = tf.py_func(
                lambda path: np.load(path.decode()).astype(np.float32),
                [path],
                [tf.float32])

            return tf.to_float(data)

    def initialize(self,
                   session,
                   image_composite_paths=None,
                   input_depth_paths=None,
                   sparse_depth_paths=None,
                   intrinsics_paths=None,
                   depth_load_multiplier=256.0,
                   do_center_crop=False,
                   do_bottom_crop=False,
                   random_horizontal_crop=False,
                   random_vertical_crop=False):

        assert session is not None

        feed_dict = {
            self.image_composite_placeholder        : image_composite_paths,
            self.input_depth_placeholder            : input_depth_paths,
            self.sparse_depth_placeholder           : sparse_depth_paths,
            self.intrinsics_placeholder             : intrinsics_paths,
            self.depth_load_multiplier_placeholder  : depth_load_multiplier,
            self.center_crop_placeholder            : do_center_crop,
            self.bottom_crop_placeholder            : do_bottom_crop,
            self.random_horizontal_crop_placeholder : random_horizontal_crop,
            self.random_vertical_crop_placeholder   : random_vertical_crop
        }

        session.run(self.iterator.initializer, feed_dict)


if __name__ == '__main__':
    import sys, os

    # Testing dataloader on KITTI
    kitti_train_image_filepath = \
        os.path.join('training', 'kitti_train_image.txt')
    kitti_train_input_depth_filepath = \
        os.path.join('training', 'kitti_train_predict_depth.txt')
    kitti_train_sparse_depth_filepath = \
        os.path.join('training', 'kitti_train_sparse_depth.txt')
    kitti_train_intrinsics_filepath = \
        os.path.join('training', 'kitti_train_intrinsics.txt')

    kitti_train_image_paths = \
        data_utils.read_paths(kitti_train_image_filepath)
    kitti_train_input_depth_paths = \
        data_utils.read_paths(kitti_train_input_depth_filepath)
    kitti_train_sparse_depth_paths = \
        data_utils.read_paths(kitti_train_sparse_depth_filepath)
    kitti_train_intrinsics_paths = \
        data_utils.read_paths(kitti_train_intrinsics_filepath)

    n_height = 320
    n_width = 768

    dataloader = FusionNetDataloader(
        name='fusionnet_dataloader',
        shape=[1, n_height, n_width, 1],
        normalize=True)

    session = tf.Session()
    dataloader.initialize(
        session,
        image_composite_paths=kitti_train_image_paths,
        input_depth_paths=kitti_train_input_depth_paths,
        sparse_depth_paths=kitti_train_sparse_depth_paths,
        intrinsics_paths=kitti_train_intrinsics_paths,
        do_center_crop=False,
        do_bottom_crop=True,
        random_horizontal_crop=True,
        random_vertical_crop=False)

    n_sample = 0
    print('Testing dataloader KITTI using paths from: \n {} \n {} \n {} \n {}'.format(
        kitti_train_image_filepath,
        kitti_train_input_depth_filepath,
        kitti_train_sparse_depth_filepath,
        kitti_train_intrinsics_filepath))

    while True:
        try:
            image0, image1, image2, input_depth, intrinsics = \
                session.run(dataloader.next_element)

            # Test shapes
            assert(image0.shape == (1, n_height, n_width, 3)), \
                'Path={} Image=0  Shape={}'.format(kitti_train_image_paths[n_sample], image0.shape)
            assert(image1.shape == (1, n_height, n_width, 3)), \
                'Path={} Image=1  Shape={}'.format(kitti_train_image_paths[n_sample], image1.shape)
            assert(image2.shape == (1, n_height, n_width, 3)), \
                'Path={} Image=2  Shape={}'.format(kitti_train_image_paths[n_sample], image2.shape)
            assert(input_depth.shape == (1, n_height, n_width, 2)), \
                'Path={} Shape={}'.format(kitti_train_input_depth_paths[n_sample], input_depth.shape)
            assert(intrinsics.shape == (1, 3, 3)), \
                'Path={} Shape={}'.format(kitti_train_intrinsics_paths[n_sample], intrinsics.shape)
            # Test values
            assert(np.min(image0) >= 0.0), \
                'Path={} Image=0  Min={}'.format(kitti_train_image_paths[n_sample], np.min(image0))
            assert(np.max(image0) <= 1.0), \
                'Path={} Image=0  Max={}'.format(kitti_train_image_paths[n_sample], np.max(image0))
            assert(np.min(image1) >= 0.0), \
                'Path={} Image=1  Min={}'.format(kitti_train_image_paths[n_sample], np.min(image1))
            assert(np.max(image1) <= 1.0), \
                'Path={} Image=1  Max={}'.format(kitti_train_image_paths[n_sample], np.max(image1))
            assert(np.min(image2) >= 0.0), \
                'Path={} Image=2  Min={}'.format(kitti_train_image_paths[n_sample], np.min(image2))
            assert(np.max(image2) <= 1.0), \
                'Path={} Image=2  Max={}'.format(kitti_train_image_paths[n_sample], np.max(image2))
            assert(np.min(input_depth[..., 0]) >= 0.0), \
                'Path={}  Min={}'.format(kitti_train_input_depth_paths[n_sample], np.min(input_depth[..., 0]))
            assert(np.max(input_depth[..., 0]) <= 256.0), \
                'Path={}  Max={}'.format(kitti_train_input_depth_paths[n_sample], np.max(input_depth[..., 0]))

            assert(np.min(input_depth[..., 1]) >= 0.0), \
                'Path={}  Min={}'.format(kitti_train_sparse_depth_paths[n_sample], np.min(input_depth[..., 1]))
            assert(np.max(input_depth[..., 1]) <= 256.0), \
                'Path={}  Max={}'.format(kitti_train_sparse_depth_paths[n_sample], np.max(input_depth[..., 1]))

            n_sample = n_sample + 1

            sys.stdout.write('Processed {} samples...\r'.format(n_sample))
            sys.stdout.flush()

        except tf.errors.OutOfRangeError:
            break

    print('Completed tests for dataloader on KITTI using {} samples'.format(n_sample))
