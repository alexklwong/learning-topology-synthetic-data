import os, sys, argparse
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

sys.path.insert(0, 'src')
import global_constants as settings
import data_utils
from scaffnet_dataloader import ScaffNetDataloader
from scaffnet_model import ScaffNetModel


parser = argparse.ArgumentParser()

# Model path
parser.add_argument('--restore_path',
    type=str, required=True, help='Model checkpoint restore path')
# Input paths
parser.add_argument('--train_sparse_depth_path',
    type=str, required=True, help='Paths to training sparse depth paths')
parser.add_argument('--train_validity_map_path',
    type=str, required=True, help='Paths to training validity map paths')
parser.add_argument('--val_sparse_depth_path',
    type=str, required=True, help='Paths to validation sparse depth paths')
parser.add_argument('--val_validity_map_path',
    type=str, required=True, help='Paths to validation validity map paths')
parser.add_argument('--test_sparse_depth_path',
    type=str, required=True, help='Paths to testing sparse depth paths')
parser.add_argument('--test_validity_map_path',
    type=str, required=True, help='Paths to testing validity map paths')
# Root directories
parser.add_argument('--input_root_dirpath',
    type=str, default='', help='Root directory for input paths')
parser.add_argument('--output_root_dirpath',
    type=str, default='', help='Root directory for output paths, used to replace root of input paths')
# Dataloader settings
parser.add_argument('--depth_load_multiplier',
    type=float, default=settings.DEPTH_LOAD_MULTIPLIER, help='Multiplier used for loading depth')
parser.add_argument('--min_dataset_depth',
    type=float, default=settings.MIN_DATASET_DEPTH, help='Minimum depth value for dataset')
parser.add_argument('--max_dataset_depth',
    type=float, default=settings.MAX_DATASET_DEPTH, help='Maximum depth value for dataset')
# Network architecture
parser.add_argument('--network_type',
    type=str, default=settings.NETWORK_TYPE_SCAFFNET, help='Network type to build')
parser.add_argument('--activation_func',
    type=str, default=settings.ACTIVATION_FUNC, help='Activation function for network')
parser.add_argument('--output_func',
    type=str, default=settings.OUTPUT_FUNC_SCAFFNET, help='Output function for network')
parser.add_argument('--n_filter_output',
    type=int, default=settings.N_FILTER_OUTPUT_SCAFFNET, help='Number of filters to use in final full resolution output')
# Spatial pyramid pooling
parser.add_argument('--pool_kernel_sizes_spp',
    nargs='+', type=int, default=settings.POOL_KERNEL_SIZES_SPP, help='Kernel sizes for spatial pyramid pooling')
parser.add_argument('--n_convolution_spp',
    type=int, default=settings.N_CONVOLUTION_SPP, help='Number of convolutions to use to balance density vs. detail tradeoff')
# Depth prediction settings
parser.add_argument('--min_predict_depth',
    type=float, default=settings.MIN_PREDICT_DEPTH, help='Minimum depth value to predict')
parser.add_argument('--max_predict_depth',
    type=float, default=settings.MAX_PREDICT_DEPTH, help='Maximum depth value to predict')
# Output options
parser.add_argument('--train_output_depth_path',
    type=str, default=os.path.join('training', 'kitti_train_predict_depth.txt'), help='Training reference paths')
parser.add_argument('--val_output_depth_path',
    type=str, default=os.path.join('validation', 'kitti_val_predict_depth.txt'), help='Validation reference paths')
parser.add_argument('--test_output_depth_path',
    type=str, default=os.path.join('testing', 'kitti_test_predict_depth.txt'), help='Testing reference paths')
# Hardware settings
parser.add_argument('--n_thread',
    type=int, default=settings.N_THREAD, help='Number of threads to use')

args = parser.parse_args()


'''
Setup for training set
'''
# Load training sparse depth and validity map paths from file
train_sparse_depth_paths = data_utils.read_paths(args.train_sparse_depth_path)
train_validity_map_paths = data_utils.read_paths(args.train_validity_map_path)

n_sample = len(train_sparse_depth_paths)

assert n_sample == len(train_validity_map_paths)

# Get shapes of images in dataset so we can update shape of dataloader
shapes = []
steps = []
for idx in range(n_sample):

    print('Read {}/{} training examples'.format(idx, n_sample), end='\r')

    sparse_depth = data_utils.load_depth(train_sparse_depth_paths[idx])

    if len(shapes) > 0:
        if sparse_depth.shape != shapes[-1]:
            shapes.append(sparse_depth.shape)
            steps.append(idx)
    else:
        shapes.append(sparse_depth.shape)
        steps.append(0)

print('Displaying step and input shape sizes for training set:')
for shape, step in zip(shapes, steps):
    print('step={} : shape={}'.format(step, shape))

print('Generating ScaffNet predictions using: {}'.format(args.restore_path))

# Predict dense depth from sparse depth
train_output_depth_paths = []
n = 0
for idx in range(len(steps)):
    shape = shapes[idx]
    step = steps[idx]

    with tf.Graph().as_default():
        # Initialize dataloader
        dataloader = ScaffNetDataloader(
            shape=[1, shape[0], shape[1], 2],
            name='scaffnet_dataloader',
            n_thread=args.n_thread,
            prefetch_size=(2 * args.n_thread))

        # Fetch the input from dataloader
        input_depth = dataloader.next_element[0]

        # Build computation graph
        model = ScaffNetModel(
            input_depth,
            input_depth,
            is_training=False,
            network_type=args.network_type,
            activation_func=args.activation_func,
            output_func=args.output_func,
            n_filter_output=args.n_filter_output,
            pool_kernel_sizes_spp=args.pool_kernel_sizes_spp,
            n_convolution_spp=args.n_convolution_spp,
            min_dataset_depth=args.min_dataset_depth,
            max_dataset_depth=args.max_dataset_depth,
            min_predict_depth=args.min_predict_depth,
            max_predict_depth=args.max_predict_depth)

        # Initialize Tensorflow session
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        session = tf.Session(config=config)

        # Load from checkpoint
        train_saver = tf.train.Saver()
        session.run(tf.global_variables_initializer())
        session.run(tf.local_variables_initializer())
        train_saver.restore(session, args.restore_path)

        if idx < len(steps) - 1:
            sparse_depth_paths = train_sparse_depth_paths[step:steps[idx + 1]]
            validity_map_paths = train_validity_map_paths[step:steps[idx + 1]]
        else:
            sparse_depth_paths = train_sparse_depth_paths[step:]
            validity_map_paths = train_validity_map_paths[step:]

        # Load sparse depth and validity maps
        dataloader.initialize(
            session,
            sparse_depth_paths=sparse_depth_paths,
            validity_map_paths=validity_map_paths,
            ground_truth_paths=sparse_depth_paths,
            depth_load_multiplier=args.depth_load_multiplier,
            do_crop=False,
            random_horizontal_crop=False,
            random_vertical_crop=False,
            random_horizontal_flip=False)

        while True:
            try:
                # Forward through network
                output_depth = np.squeeze(session.run(model.predict))

                assert train_validity_map_paths[n].find('validity_map'), train_validity_map_paths[n]
                output_depth_path = train_validity_map_paths[n].replace('validity_map', 'prediction')

                # Replace input root directory with output root directory
                if args.input_root_dirpath != '' and args.output_root_dirpath != '':
                    output_depth_path = output_depth_path \
                        .replace(args.input_root_dirpath, args.output_root_dirpath)

                output_depth_dirpath = os.path.dirname(output_depth_path)
                if not os.path.exists(output_depth_dirpath):
                    os.makedirs(output_depth_dirpath)

                train_output_depth_paths.append(output_depth_path)
                data_utils.save_depth(output_depth, output_depth_path)

                n += 1
                print('Processed {}/{} training examples \r'.format(n + 1, n_sample), end='\r')

            except tf.errors.OutOfRangeError:
                print('Currently processed {}/{} training examples'.format(n, n_sample))
                break

print('Storing prediction for training file paths into: %s' % args.train_output_depth_path)
data_utils.write_paths(args.train_output_depth_path, train_output_depth_paths)


'''
Setup for validation and testing set
'''
# Load validation sparse depth and validity map paths from file
val_sparse_depth_paths = data_utils.read_paths(args.val_sparse_depth_path)
val_validity_map_paths = data_utils.read_paths(args.val_validity_map_path)

assert len(val_sparse_depth_paths) == len(val_validity_map_paths)

# Load testing sparse depth and validity map paths from file
test_sparse_depth_paths = data_utils.read_paths(args.test_sparse_depth_path)
test_validity_map_paths = data_utils.read_paths(args.test_validity_map_path)

assert len(test_sparse_depth_paths) == len(test_validity_map_paths)

val_output_depth_paths = []
test_output_depth_paths = []
modes = [
    [
        'validation',
        val_sparse_depth_paths,
        val_validity_map_paths,
        val_output_depth_paths
    ], [
        'testing',
        test_sparse_depth_paths,
        test_validity_map_paths,
        test_output_depth_paths
    ]
]

for mode in modes:
    n = 0
    mode_type, sparse_depth_paths, validity_map_paths, output_depth_paths = mode
    shape = np.squeeze(data_utils.load_depth(sparse_depth_paths[0])).shape

    n_sample = len(sparse_depth_paths)
    with tf.Graph().as_default():
        # Initialize dataloader
        dataloader = ScaffNetDataloader(
            shape=[1, shape[0], shape[1], 2],
            name='scaffnet_dataloader',
            n_thread=args.n_thread,
            prefetch_size=(2 * args.n_thread))

        # Fetch the input from dataloader
        input_depth = dataloader.next_element[0]

        # Build computation graph
        model = ScaffNetModel(
            input_depth,
            input_depth,
            is_training=False,
            network_type=args.network_type,
            activation_func=args.activation_func,
            output_func=args.output_func,
            n_filter_output=args.n_filter_output,
            pool_kernel_sizes_spp=args.pool_kernel_sizes_spp,
            n_convolution_spp=args.n_convolution_spp,
            min_dataset_depth=args.min_dataset_depth,
            max_dataset_depth=args.max_dataset_depth,
            min_predict_depth=args.min_predict_depth,
            max_predict_depth=args.max_predict_depth)

        # Initialize Tensorflow session
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        session = tf.Session(config=config)

        # Load from checkpoint
        train_saver = tf.train.Saver()
        session.run(tf.global_variables_initializer())
        session.run(tf.local_variables_initializer())
        train_saver.restore(session, args.restore_path)

        # Load sparse depth and valid maps
        dataloader.initialize(
            session,
            sparse_depth_paths=sparse_depth_paths,
            validity_map_paths=validity_map_paths,
            ground_truth_paths=sparse_depth_paths,
            depth_load_multiplier=args.depth_load_multiplier,
            do_crop=False,
            random_horizontal_crop=False,
            random_vertical_crop=False,
            random_horizontal_flip=False)

        while True:
            try:
                output_depth = np.squeeze(session.run(model.predict))

                output_depth_path = validity_map_paths[n].replace('validity_map', 'prediction')

                # Replace input root directory with output root directory
                if args.input_root_dirpath != '' and args.output_root_dirpath != '':
                    output_depth_path = output_depth_path \
                        .replace(args.input_root_dirpath, args.output_root_dirpath)

                output_depth_dirpath = os.path.dirname(output_depth_path)
                if not os.path.exists(output_depth_dirpath):
                    os.makedirs(output_depth_dirpath)

                output_depth_paths.append(output_depth_path)
                data_utils.save_depth(output_depth, output_depth_path)

                n += 1
                print('Processed {}/{} {} examples \r'.format(n + 1, n_sample, mode_type), end='\r')

            except tf.errors.OutOfRangeError:
                print('Finished processing {}/{} {} examples'.format(n, n_sample, mode_type))
                break

print('Storing prediction for validation file paths into: %s' % args.val_output_depth_path)
data_utils.write_paths(args.val_output_depth_path, val_output_depth_paths)

print('Storing prediction for testing file paths into: %s' % args.test_output_depth_path)
data_utils.write_paths(args.test_output_depth_path, test_output_depth_paths)
