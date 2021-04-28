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

            # Set up placeholder for loading depth
            self.depth_load_multiplier_placeholder = tf.placeholder(tf.float32, shape=())

            # Set up crop and data augmentation placeholders
            self.center_crop_placeholder = tf.placeholder(tf.bool, shape=())
            self.bottom_crop_placeholder = tf.placeholder(tf.bool, shape=())

            self.dataset = tf.data.Dataset.from_tensor_slices((
                self.image_placeholder,
                self.sparse_depth_placeholder))

            self.dataset = self.dataset \
                .map(self._load_func, num_parallel_calls=self.n_thread) \
                .map(self._crop_func, num_parallel_calls=self.n_thread) \
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

    def _crop_func(self, image, input_depth):
        '''
        Crops image and input depth to specified shape

        Args:
            image : tensor
                H x W x 3 RGB image
            input_depth : tensor
                H x W x 2 input depth of sparse depth and validity map
        Returns:
            tensor : h x w x 3 RGB image
            tensor : h x w x 2 sparse depth and validity map
        '''

        def crop_func(in0, in1):
            # Center crop to specified height and width, default bottom centered
            shape = tf.shape(in0)

            start_height = tf.cond(
                self.center_crop_placeholder,
                lambda: tf.to_int32(tf.to_float(shape[0] - self.n_height) / tf.to_float(2.0)),
                lambda: tf.to_int32(shape[0] - self.n_height))

            end_height = self.n_height + start_height

            start_width = tf.to_int32(tf.to_float(shape[1] - self.n_width) / tf.to_float(2.0))
            end_width = self.n_width + start_width

            # Apply crop
            in0 = in0[start_height:end_height, start_width:end_width, :]
            in1 = in1[start_height:end_height, start_width:end_width, :]

            return in0, in1

        with tf.variable_scope('crop_func'):
            image, input_depth = tf.cond(
                tf.math.logical_or(self.center_crop_placeholder, self.bottom_crop_placeholder),
                lambda: crop_func(image, input_depth),
                lambda: (image, input_depth))

            return (image, input_depth)

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
            intrinsics = tf.cond(
                tf.equal(path, ''),
                lambda : tf.eye(3),
                lambda : tf.reshape(self._load_npy_func(path), [3, 3]))

            return intrinsics

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
                   image_paths=None,
                   sparse_depth_paths=None,
                   depth_load_multiplier=256.0,
                   load_image_composite=True,
                   do_center_crop=False,
                   do_bottom_crop=False):

        assert session is not None

        feed_dict = {
            self.image_placeholder                  : image_paths,
            self.sparse_depth_placeholder           : sparse_depth_paths,
            self.depth_load_multiplier_placeholder  : depth_load_multiplier,
            self.load_image_composite_placeholder   : load_image_composite,
            self.center_crop_placeholder            : do_center_crop,
            self.bottom_crop_placeholder            : do_bottom_crop,
        }

        session.run(self.iterator.initializer, feed_dict)


if __name__ == '__main__':
    import sys, os

    # Testing dataloader on KITTI
    kitti_train_image_filepath = \
        os.path.join('training', 'kitti_train_image.txt')
    kitti_train_sparse_depth_filepath = \
        os.path.join('training', 'kitti_train_sparse_depth.txt')

    kitti_train_image_paths = \
        data_utils.read_paths(kitti_train_image_filepath)
    kitti_train_sparse_depth_paths = \
        data_utils.read_paths(kitti_train_sparse_depth_filepath)

    n_height = 320
    n_width = 768

    dataloader = FusionNetStandaloneDataloader(
        name='fusionnet_standalone_dataloader',
        shape=[1, n_height, n_width, 1],
        normalize=True)

    session = tf.Session()
    dataloader.initialize(
        session,
        image_composite_paths=kitti_train_image_paths,
        sparse_depth_paths=kitti_train_sparse_depth_paths,
        depth_load_multiplier=256.0,
        load_image_composite=True,
        do_center_crop=False,
        do_bottom_crop=True)

    n_sample = 0
    print('Testing dataloader KITTI using paths from: \n {} \n {}'.format(
        kitti_train_image_filepath,
        kitti_train_sparse_depth_filepath))

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
