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
from fusionnet import train


parser = argparse.ArgumentParser()

# Training and validation input filepaths
parser.add_argument('--train_image_path',
    type=str, required=True, help='Paths to training image paths')
parser.add_argument('--train_input_depth_path',
    type=str, required=True, help='Paths to training input (predicted) depth paths')
parser.add_argument('--train_sparse_depth_path',
    type=str, required=True, help='Paths to training sparse depth paths')
parser.add_argument('--train_intrinsics_path',
    type=str, required=True, help='Paths to training intrinsics paths')
parser.add_argument('--val_image_path',
    type=str, default='', help='Paths to validation image composite paths')
parser.add_argument('--val_input_depth_path',
    type=str, default='', help='Paths to validation input (predicted) depth paths')
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
parser.add_argument('--crop_type',
    type=str, default=settings.CROP_TYPE, help='Crop to perform when loading data')
parser.add_argument('--augmentation_random_horizontal_crop',
    action='store_true', help='If set, perform random crop in horizontal direction for augmentation')
parser.add_argument('--augmentation_random_vertical_crop',
    action='store_true', help='If set, perform random crop in vertical direction for augmentation')
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
# Training settings
parser.add_argument('--learning_rates',
    nargs='+', type=float, default=settings.LEARNING_RATES, help='Comma delimited learning rates')
parser.add_argument('--learning_schedule',
    nargs='+', type=int, default=settings.LEARNING_SCHEDULE, help='Comma delimited learning schedule')
parser.add_argument('--n_epoch',
    type=int, default=settings.N_EPOCH, help='Total number of epochs to train')
# Loss function settings
parser.add_argument('--validity_map_color',
    type=str, default=settings.VALIDITY_MAP_COLOR, help='Determines where to compute photometric loss')
parser.add_argument('--w_color',
    type=float, default=settings.W_COLOR, help='Weight of color consistency')
parser.add_argument('--w_structure',
    type=float, default=settings.W_STRUCTURE, help='Weight of structural consistency')
parser.add_argument('--w_sparse_depth',
    type=float, default=settings.W_SPARSE_DEPTH, help='Weight of sparse depth consistency')
parser.add_argument('--w_smoothness',
    type=float, default=settings.W_SMOOTHNESS, help='Weight of local smoothness')
parser.add_argument('--w_prior_depth',
    type=float, default=settings.W_PRIOR_DEPTH, help='Weight of topology prior')
parser.add_argument('--rotation_param',
    type=str, default=settings.ROTATION_PARAM, help='Rotation parameterization')
parser.add_argument('--residual_threshold_prior_depth',
    type=float, default=settings.RESIDUAL_THRESHOLD_PRIOR_DEPTH, help='Residual threshold to use depth prior')
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

    args.val_image_path = \
        None if args.val_image_path == '' else args.val_image_path
    args.val_input_depth_path = \
        None if args.val_input_depth_path == '' else args.val_input_depth_path
    args.val_sparse_depth_path = \
        None if args.val_sparse_depth_path == '' else args.val_sparse_depth_path
    args.val_ground_truth_path = \
        None if args.val_ground_truth_path == '' else args.val_ground_truth_path

    train(train_image_path=args.train_image_path,
          train_input_depth_path=args.train_input_depth_path,
          train_sparse_depth_path=args.train_sparse_depth_path,
          train_intrinsics_path=args.train_intrinsics_path,
          # Validation data filepaths
          val_image_path=args.val_image_path,
          val_input_depth_path=args.val_input_depth_path,
          val_sparse_depth_path=args.val_sparse_depth_path,
          val_ground_truth_path=args.val_ground_truth_path,
          # Dataloader settings
          n_batch=args.n_batch,
          n_height=args.n_height,
          n_width=args.n_width,
          crop_type=args.crop_type,
          augmentation_random_horizontal_crop=args.augmentation_random_horizontal_crop,
          augmentation_random_vertical_crop=args.augmentation_random_vertical_crop,
          # Network settings
          network_type=args.network_type,
          image_filter_pct=args.image_filter_pct,
          depth_filter_pct=args.depth_filter_pct,
          activation_func=args.activation_func,
          # Depth prediction settings
          min_predict_depth=args.min_predict_depth,
          max_predict_depth=args.max_predict_depth,
          min_scale_depth=args.min_scale_depth,
          max_scale_depth=args.max_scale_depth,
          min_residual_depth=args.min_residual_depth,
          max_residual_depth=args.max_residual_depth,
          # Training settings
          n_epoch=args.n_epoch,
          learning_rates=args.learning_rates,
          learning_schedule=args.learning_schedule,
          # Loss function settings
          validity_map_color=args.validity_map_color,
          w_color=args.w_color,
          w_structure=args.w_structure,
          w_sparse_depth=args.w_sparse_depth,
          w_smoothness=args.w_smoothness,
          w_prior_depth=args.w_prior_depth,
          rotation_param=args.rotation_param,
          residual_threshold_prior_depth=args.residual_threshold_prior_depth,
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
