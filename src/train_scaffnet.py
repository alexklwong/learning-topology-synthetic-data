import argparse
import global_constants as settings
from scaffnet import train


parser = argparse.ArgumentParser()

# Training and validation input filepaths
parser.add_argument('--train_sparse_depth_path',
    type=str, required=True, help='Paths to training sparse depth paths')
parser.add_argument('--train_validity_map_path',
    type=str, required=True, help='Paths to training validity map paths')
parser.add_argument('--train_ground_truth_path',
    type=str, required=True, help='Paths to training ground truth paths')
parser.add_argument('--val_sparse_depth_path',
    type=str, default='', help='Paths to validation sparse depth paths')
parser.add_argument('--val_validity_map_path',
    type=str, default='', help='Paths to validation validity map paths')
parser.add_argument('--val_ground_truth_path',
    type=str, default='', help='Paths to validation ground truth paths')
# Dataloader settings
parser.add_argument('--depth_load_multiplier',
    type=float, default=settings.DEPTH_LOAD_MULTIPLIER, help='Multiplier used for loading depth')
parser.add_argument('--min_dataset_depth',
    type=float, default=settings.MIN_DATASET_DEPTH, help='Minimum depth value for dataset')
parser.add_argument('--max_dataset_depth',
    type=float, default=settings.MAX_DATASET_DEPTH, help='Maximum depth value for dataset')
parser.add_argument('--augment_random_crop',
    action='store_true', help='If set, perform random crop for augmentation')
parser.add_argument('--augment_random_horizontal_flip',
    action='store_true', help='If set, perform horizontal and vertical flip augmentation')
# Batch parameters
parser.add_argument('--n_batch',
    type=int, default=settings.N_BATCH, help='Number of samples per batch')
parser.add_argument('--n_height',
    type=int, default=settings.N_HEIGHT, help='Height of each sample')
parser.add_argument('--n_width',
    type=int, default=settings.N_WIDTH, help='Width of each sample')
# Training parameters
parser.add_argument('--learning_rates',
    nargs='+', type=float, default=settings.LEARNING_RATES, help='Comma delimited learning rates')
parser.add_argument('--learning_schedule',
    nargs='+', type=int, default=settings.LEARNING_SCHEDULE, help='Comma delimited learning schedule')
parser.add_argument('--n_epoch',
    type=int, default=settings.N_EPOCH, help='Total number of epochs to train')
parser.add_argument('--loss_func',
    type=str, default=settings.LOSS_FUNC_SCAFFNET, help='Loss function to use')
# Network architecture
parser.add_argument('--network_type',
    type=str, default=settings.NETWORK_TYPE_SCAFFNET, help='Network type to build')
parser.add_argument('--activation_func',
    type=str, default=settings.ACTIVATION_FUNC, help='Activation function for network')
parser.add_argument('--output_func',
    type=str, default=settings.OUTPUT_FUNC, help='Output function for network')
parser.add_argument('--n_filter_output',
    type=int, default=settings.N_FILTER_OUTPUT, help='Number of filters to use in final full resolution output')
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
# Depth evaluation settings
parser.add_argument('--min_evaluate_depth',
    type=float, default=settings.MIN_EVALUATE_DEPTH, help='Minimum depth value evaluate')
parser.add_argument('--max_evaluate_depth',
    type=float, default=settings.MAX_EVALUATE_DEPTH, help='Maximum depth value to evaluate')
# Checkpoint and restore paths
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
    args.val_validity_map_path = \
        None if args.val_validity_map_path == '' else args.val_validity_map_path
    args.val_ground_truth_path = \
        None if args.val_ground_truth_path == '' else args.val_ground_truth_path

    train(train_sparse_depth_path=args.train_sparse_depth_path,
          train_validity_map_path=args.train_validity_map_path,
          train_ground_truth_path=args.train_ground_truth_path,
          # Validation data
          val_sparse_depth_path=args.val_sparse_depth_path,
          val_validity_map_path=args.val_validity_map_path,
          val_ground_truth_path=args.val_ground_truth_path,
          # Dataloader settings
          depth_load_multiplier=args.depth_load_multiplier,
          min_dataset_depth=args.min_dataset_depth,
          max_dataset_depth=args.max_dataset_depth,
          augment_random_crop=args.augment_random_crop,
          augment_random_horizontal_flip=args.augment_random_horizontal_flip,
          # Batch parameters
          n_batch=args.n_batch,
          n_height=args.n_height,
          n_width=args.n_width,
          # Training settings
          learning_rates=args.learning_rates,
          learning_schedule=args.learning_schedule,
          n_epoch=args.n_epoch,
          loss_func=args.loss_func,
          # Network architecture
          network_type=args.network_type,
          activation_func=args.activation_func,
          output_func=args.output_func,
          n_filter_output=args.n_filter_output,
          # Spatial pyramid pooling
          pool_kernel_sizes_spp=args.pool_kernel_sizes_spp,
          n_convolution_spp=args.n_convolution_spp,
          # Depth prediction settings
          min_predict_depth=args.min_predict_depth,
          max_predict_depth=args.max_predict_depth,
          # Depth evaluation settings
          min_evaluate_depth=args.min_evaluate_depth,
          max_evaluate_depth=args.max_evaluate_depth,
          # Model checkpoints
          n_checkpoint=args.n_checkpoint,
          n_summary=args.n_summary,
          checkpoint_path=args.checkpoint_path,
          restore_path=args.restore_path,
          # Hardware settings
          n_thread=args.n_thread)
