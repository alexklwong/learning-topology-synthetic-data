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
import argparse
import global_constants as settings
from scaffnet import train


parser = argparse.ArgumentParser()

# Training and validation input filepaths
parser.add_argument('--train_sparse_depth_path',
    type=str, required=True, help='Paths to training sparse depth paths')
parser.add_argument('--train_ground_truth_path',
    type=str, required=True, help='Paths to training ground truth paths')
parser.add_argument('--val_sparse_depth_path',
    type=str, default='', help='Paths to validation sparse depth paths')
parser.add_argument('--val_ground_truth_path',
    type=str, default='', help='Paths to validation ground truth paths')
# Dataloader settings
parser.add_argument('--n_batch',
    type=int, default=settings.N_BATCH, help='Number of samples per batch')
parser.add_argument('--n_height',
    type=int, default=settings.N_HEIGHT, help='Height of each sample')
parser.add_argument('--n_width',
    type=int, default=settings.N_WIDTH, help='Width of each sample')
parser.add_argument('--min_dataset_depth',
    type=float, default=settings.MIN_DATASET_DEPTH, help='Minimum depth value for dataset')
parser.add_argument('--max_dataset_depth',
    type=float, default=settings.MAX_DATASET_DEPTH, help='Maximum depth value for dataset')
parser.add_argument('--augmentation_random_horizontal_crop',
    action='store_true', help='If set, perform random crop in horizontal direction for augmentation')
parser.add_argument('--augmentation_random_vertical_crop',
    action='store_true', help='If set, perform random crop in vertical direction for augmentation')
parser.add_argument('--augmentation_random_horizontal_flip',
    action='store_true', help='If set, perform horizontal and vertical flip augmentation')
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
# Training parameters
parser.add_argument('--learning_rates',
    nargs='+', type=float, default=settings.LEARNING_RATES, help='Comma delimited learning rates')
parser.add_argument('--learning_schedule',
    nargs='+', type=int, default=settings.LEARNING_SCHEDULE, help='Comma delimited learning schedule')
parser.add_argument('--n_epoch',
    type=int, default=settings.N_EPOCH, help='Total number of epochs to train')
parser.add_argument('--loss_func',
    type=str, default=settings.LOSS_FUNC_SCAFFNET, help='Loss function to use')
# Depth evaluation settings
parser.add_argument('--min_evaluate_depth',
    type=float, default=settings.MIN_EVALUATE_DEPTH, help='Minimum depth value evaluate')
parser.add_argument('--max_evaluate_depth',
    type=float, default=settings.MAX_EVALUATE_DEPTH, help='Maximum depth value to evaluate')
# Checkpoint settings
parser.add_argument('--checkpoint_path',
    type=str, default=settings.CHECKPOINT_PATH, help='Path to save checkpoints')
parser.add_argument('--restore_path',
    type=str, default=settings.RESTORE_PATH, help='Path of model to restore')
parser.add_argument('--n_checkpoint',
    type=int, default=settings.N_CHECKPOINT, help='Number of steps before saving a checkpoint')
parser.add_argument('--n_summary',
    type=int, default=settings.N_SUMMARY, help='Number of steps before logging summary')
# Hardware settings
parser.add_argument('--n_thread',
    type=int, default=settings.N_THREAD, help='Number of threads to use')

args = parser.parse_args()


if __name__ == '__main__':

    assert len(args.learning_rates) == (len(args.learning_schedule) + 1)

    args.val_sparse_depth_path = \
        None if args.val_sparse_depth_path == '' else args.val_sparse_depth_path
    args.val_ground_truth_path = \
        None if args.val_ground_truth_path == '' else args.val_ground_truth_path

    train(train_sparse_depth_path=args.train_sparse_depth_path,
          train_ground_truth_path=args.train_ground_truth_path,
          # Validation data
          val_sparse_depth_path=args.val_sparse_depth_path,
          val_ground_truth_path=args.val_ground_truth_path,
          # Dataloader settings
          n_batch=args.n_batch,
          n_height=args.n_height,
          n_width=args.n_width,
          min_dataset_depth=args.min_dataset_depth,
          max_dataset_depth=args.max_dataset_depth,
          augmentation_random_horizontal_crop=args.augmentation_random_horizontal_crop,
          augmentation_random_vertical_crop=args.augmentation_random_vertical_crop,
          augmentation_random_horizontal_flip=args.augmentation_random_horizontal_flip,
          # Network architecture
          network_type=args.network_type,
          activation_func=args.activation_func,
          n_filter_output=args.n_filter_output,
          min_predict_depth=args.min_predict_depth,
          max_predict_depth=args.max_predict_depth,
          # Spatial pyramid pooling
          pool_kernel_sizes_spp=args.pool_kernel_sizes_spp,
          n_convolution_spp=args.n_convolution_spp,
          n_filter_spp=args.n_filter_spp,
          # Training settings
          learning_rates=args.learning_rates,
          learning_schedule=args.learning_schedule,
          n_epoch=args.n_epoch,
          loss_func=args.loss_func,
          # Depth evaluation settings
          min_evaluate_depth=args.min_evaluate_depth,
          max_evaluate_depth=args.max_evaluate_depth,
          # Checkpoint settings
          n_checkpoint=args.n_checkpoint,
          n_summary=args.n_summary,
          checkpoint_path=args.checkpoint_path,
          restore_path=args.restore_path,
          # Hardware settings
          n_thread=args.n_thread)
