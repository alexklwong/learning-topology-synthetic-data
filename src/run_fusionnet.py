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
import os, time, argparse
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

import global_constants as settings
import data_utils, eval_utils
from fusionnet_dataloader import FusionNetDataloader
from fusionnet_model import FusionNetModel
from fusionnet import run
from log_utils import log


N_HEIGHT = 352
N_WIDTH = 1216


parser = argparse.ArgumentParser()

# Model path
parser.add_argument('--restore_path',
    type=str, required=True, help='FusionNet model checkpoint restore path')
# Input paths
parser.add_argument('--image_path',
    type=str, required=True, help='Paths to image paths')
parser.add_argument('--input_depth_path',
    type=str, default=True, help='Paths to input depth paths')
parser.add_argument('--sparse_depth_path',
    type=str, required=True, help='Paths to sparse depth map paths')
parser.add_argument('--ground_truth_path',
    type=str, default='', help='Paths to ground truth paths')
parser.add_argument('--load_image_composite',
    action='store_true', help='If set, load an image from image triplet composite')
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
parser.add_argument('--image_filter_pct',
    type=float, default=settings.IMAGE_FILTER_PCT, help='Percentage of filters to use for image branch')
parser.add_argument('--depth_filter_pct',
    type=float, default=settings.DEPTH_FILTER_PCT, help='Percentage of filters to use for depth branch')
parser.add_argument('--activation_func',
    type=str, default=settings.ACTIVATION_FUNC, help='Activation function for network')
# Depth prediction settings
parser.add_argument('--min_predict_depth',
    type=float, default=settings.MIN_PREDICT_DEPTH, help='Minimum depth prediction value')
parser.add_argument('--max_predict_depth',
    type=float, default=settings.MAX_PREDICT_DEPTH, help='Maximum depth prediction value')
parser.add_argument('--min_scale_depth',
    type=float, default=settings.MIN_SCALE_DEPTH, help='Minimum depth scale value')
parser.add_argument('--max_scale_depth',
    type=float, default=settings.MAX_SCALE_DEPTH, help='Maximum depth scale value')
parser.add_argument('--min_residual_depth',
    type=float, default=settings.MIN_RESIDUAL_DEPTH, help='Minimum depth residual value')
parser.add_argument('--max_residual_depth',
    type=float, default=settings.MAX_RESIDUAL_DEPTH, help='Maximum depth residual value')
# Depth evaluation settings
parser.add_argument('--min_evaluate_depth',
    type=float, default=settings.MIN_EVALUATE_DEPTH, help='Minimum depth value evaluate')
parser.add_argument('--max_evaluate_depth',
    type=float, default=settings.MAX_EVALUATE_DEPTH, help='Maximum depth value to evaluate')
# Output options
parser.add_argument('--save_outputs',
    action='store_true', help='If set, then save outputs')
parser.add_argument('--keep_input_filenames',
    action='store_true', help='If set then keep original input filenames')
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

# Load image, input depth, and sparse depth from file for evaluation
image_paths = data_utils.read_paths(args.image_path)
image_paths = image_paths[args.start_idx:args.end_idx]

input_depth_paths = data_utils.read_paths(args.input_depth_path)
input_depth_paths = input_depth_paths[args.start_idx:args.end_idx]

sparse_depth_paths = data_utils.read_paths(args.sparse_depth_path)
sparse_depth_paths = sparse_depth_paths[args.start_idx:args.end_idx]

n_sample = len(image_paths)

assert n_sample == len(input_depth_paths)
assert n_sample == len(sparse_depth_paths)

# Pad all paths based on batch size
image_paths = data_utils.pad_batch(image_paths, args.n_batch)
input_depth_paths = data_utils.pad_batch(input_depth_paths, args.n_batch)
sparse_depth_paths = data_utils.pad_batch(sparse_depth_paths, args.n_batch)

n_step = n_sample // args.n_batch

ground_truth_available = True if args.ground_truth_path != '' else False
ground_truths = []

if ground_truth_available:
    ground_truth_paths = data_utils.read_paths(args.ground_truth_path)
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
    # Initialize dataloader
    dataloader = FusionNetDataloader(
        shape=[args.n_batch, args.n_height, args.n_width, 3],
        name='fusionnet_dataloader',
        is_training=False,
        n_thread=args.n_thread,
        prefetch_size=2 * args.n_thread)

    # Fetch the input from dataloader
    image0 = dataloader.next_element[0]
    input_depth = dataloader.next_element[1]

    # Build computation graph
    model = FusionNetModel(
        image0,
        input_depth,
        is_training=False,
        network_type=args.network_type,
        image_filter_pct=args.image_filter_pct,
        depth_filter_pct=args.depth_filter_pct,
        activation_func=args.activation_func,
        min_predict_depth=args.min_predict_depth,
        max_predict_depth=args.max_predict_depth,
        min_scale_depth=args.min_scale_depth,
        max_scale_depth=args.max_scale_depth,
        min_residual_depth=args.min_residual_depth,
        max_residual_depth=args.max_residual_depth)

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

    # Load data
    dataloader.initialize(
        session,
        image_paths=image_paths,
        input_depth_paths=input_depth_paths,
        sparse_depth_paths=sparse_depth_paths,
        load_image_composite=args.load_image_composite)

    time_start = time.time()

    # Forward through the network
    output_depths = run(model, session, n_sample, verbose=True)

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

            if args.keep_input_filenames:
                filename = os.path.basename(sparse_depth_paths[idx])
            else:
                filename = '{:010d}.png'.format(idx)

            output_path = os.path.join(output_dirpath, filename)
            data_utils.save_depth(output_depth, output_path)
