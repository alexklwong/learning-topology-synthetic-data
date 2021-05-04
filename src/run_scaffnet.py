import os, time, argparse
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

import global_constants as settings
import data_utils, eval_utils
from scaffnet_dataloader import ScaffNetDataloader
from scaffnet_model import ScaffNetModel
from scaffnet import run
from log_utils import log


N_HEIGHT = 352
N_WIDTH = 1216


parser = argparse.ArgumentParser()

# Model path
parser.add_argument('--restore_path',
    type=str, required=True, help='Model checkpoint restore path')
# Input paths
parser.add_argument('--sparse_depth_path',
    type=str, required=True, help='Paths to sparse depth paths')
parser.add_argument('--ground_truth_path',
    type=str, default='', help='Paths to ground truth paths')
# Dataloader settings
parser.add_argument('--start_idx',
    type=int, default=0, help='Start of subset of samples to run')
parser.add_argument('--end_idx',
    type=int, default=1000, help='End of subset of samples to run')
# Batch parameters
parser.add_argument('--n_batch',
    type=int, default=settings.N_BATCH, help='Number of samples per batch')
parser.add_argument('--n_height',
    type=int, default=N_HEIGHT, help='Height of each sample')
parser.add_argument('--n_width',
    type=int, default=N_WIDTH, help='Width of each sample')
# Network architecture
parser.add_argument('--network_type',
    type=str, default=settings.NETWORK_TYPE_SCAFFNET, help='Network type to build')
parser.add_argument('--activation_func',
    type=str, default=settings.ACTIVATION_FUNC, help='Activation function for network')
parser.add_argument('--n_filter_output',
    type=int, default=settings.N_FILTER_OUTPUT_SCAFFNET, help='Number of filters to use in final full resolution output')
parser.add_argument('--min_predict_depth',
    type=float, default=settings.MIN_PREDICT_DEPTH, help='Minimum depth value to predict')
parser.add_argument('--max_predict_depth',
    type=float, default=settings.MAX_PREDICT_DEPTH, help='Maximum depth value to predict')
# Spatial pyramid pooling
parser.add_argument('--pool_kernel_sizes_spp',
    nargs='+', type=int, default=settings.POOL_KERNEL_SIZES_SPP, help='Kernel sizes for spatial pyramid pooling')
parser.add_argument('--n_convolution_spp',
    type=int, default=settings.N_CONVOLUTION_SPP, help='Number of convolutions to use to balance density vs. detail tradeoff')
parser.add_argument('--n_filter_spp',
    type=int, default=settings.N_FILTER_SPP, help='Number of filters to use in 1 x 1 convolutions in spatial pyramid pooling')
# Depth evaluation settings
parser.add_argument('--min_evaluate_depth',
    type=float, default=settings.MIN_EVALUATE_DEPTH, help='Minimum depth value evaluate')
parser.add_argument('--max_evaluate_depth',
    type=float, default=settings.MAX_EVALUATE_DEPTH, help='Maximum depth value to evaluate')
# Output options
parser.add_argument('--save_outputs',
    action='store_true', help='If set, then save outputs')
parser.add_argument('--output_path',
    type=str, default='output', help='Path to save outputs')
# Hardware settings
parser.add_argument('--n_thread',
    type=int, default=settings.N_THREAD, help='Number of threads to use')

args = parser.parse_args()


'''
Read input paths and load ground truth (if available)
'''
log_path = os.path.join(args.output_path, 'results.txt')

if not os.path.exists(args.output_path):
    os.makedirs(args.output_path)

# Load sparse depth and validity map paths from file for evaluation
sparse_depth_paths = sorted(data_utils.read_paths(args.sparse_depth_path))
sparse_depth_paths = sparse_depth_paths[args.start_idx:args.end_idx]

n_sample = len(sparse_depth_paths)

# Pad all paths based on batch size
sparse_depth_paths = data_utils.pad_batch(sparse_depth_paths, args.n_batch)

n_step = n_sample // args.n_batch

ground_truth_available = True if args.ground_truth_path != '' else False
ground_truths = []

if ground_truth_available:
    ground_truth_paths = sorted(data_utils.read_paths(args.ground_truth_path))
    ground_truth_paths = ground_truth_paths[args.start_idx:args.end_idx]

    assert n_sample == len(ground_truth_paths)

    # Load ground truth
    for idx in range(n_sample):

        print('Loading {}/{} groundtruth depth maps'.format(idx + 1, n_sample), end='\r')

        ground_truth, validity_map = \
            data_utils.load_depth_with_validity_map(ground_truth_paths[idx])

        ground_truth = np.concatenate([
            np.expand_dims(ground_truth, axis=-1),
            np.expand_dims(validity_map, axis=-1)], axis=-1)
        ground_truths.append(ground_truth)

    ground_truth_paths = data_utils.pad_batch(ground_truth_paths, args.n_batch)

    print('Completed loading {} groundtruth depth maps'.format(n_sample))


'''
Build graph
'''
with tf.Graph().as_default():
    dataloader = ScaffNetDataloader(
        shape=[args.n_batch, args.n_height, args.n_width, 2],
        name='scaffnet_dataloader',
        is_training=False,
        n_thread=args.n_thread,
        prefetch_size=(2 * args.n_thread))

    # Fetch the input from dataloader
    input_depth = dataloader.next_element[0]

    # Build computation graph
    scaffnet = ScaffNetModel(
        input_depth,
        is_training=False,
        network_type=args.network_type,
        activation_func=args.activation_func,
        n_filter_output=args.n_filter_output,
        pool_kernel_sizes_spp=args.pool_kernel_sizes_spp,
        n_convolution_spp=args.n_convolution_spp,
        n_filter_spp=args.n_filter_spp,
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

    log('Evaluating {}'.format(args.restore_path), log_path)

    # Load sparse depth and valid maps
    dataloader.initialize(
        session,
        sparse_depth_paths=sparse_depth_paths)

    time_start = time.time()

    # Forward through the network
    output_depths = run(scaffnet, session, n_sample, verbose=True)

    # Measure run time
    time_elapse = time.time() - time_start
    time_elapse_per_sample = time_elapse / float(output_depths.shape[0])

    log('Total time: {:.2f} min  Average time per sample: {:.2f} ms'.format(
        time_elapse / 60.0, time_elapse_per_sample * 1000.0),
        log_path)

    if ground_truth_available:
        # Run evaluation metrics
        eval_utils.evaluate(
            output_depths,
            ground_truths,
            log_path=log_path,
            min_evaluate_depth=args.min_evaluate_depth,
            max_evaluate_depth=args.max_evaluate_depth)

    # Store output depth as images
    if args.save_outputs:
        output_dirpath = os.path.join(args.output_path, 'saved')
        print('Storing output depth as PNG into {}'.format(output_dirpath))

        if not os.path.exists(output_dirpath):
            os.makedirs(output_dirpath)

        for idx in range(n_sample):
            output_depth = np.squeeze(output_depths[idx, ...])
            _, filename = os.path.split(sparse_depth_paths[idx])
            output_path = os.path.join(output_dirpath, filename)
            data_utils.save_depth(output_depth, output_path)
