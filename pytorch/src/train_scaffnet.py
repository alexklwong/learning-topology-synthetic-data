import argparse
from scaffnet_main import train


parser = argparse.ArgumentParser()

# Training and validation input filepaths
parser.add_argument('--train_sparse_depth_path',
    type=str, required=True, help='Path to list of training sparse depth paths')
parser.add_argument('--train_ground_truth_path',
    type=str, required=True, help='Path to list of training groundtruth depth paths')
parser.add_argument('--val_sparse_depth_path',
    type=str, default=None, help='Path to list of validation sparse depth paths')
parser.add_argument('--val_ground_truth_path',
    type=str, default=None, help='Path to list of validation ground truth depth paths')

# Batch settings
parser.add_argument('--n_batch',
    type=int, default=8, help='Number of samples per batch')
parser.add_argument('--n_height',
    type=int, default=240, help='Height of of sample')
parser.add_argument('--n_width',
    type=int, default=320, help='Width of each sample')

# Dataset settings
parser.add_argument('--cap_dataset_depth_method',
    type=str, default='set_to_max', help='Method to cap depth')
parser.add_argument('--min_dataset_depth',
    type=float, default=0.0, help='Minimum value of dataset depth')
parser.add_argument('--max_dataset_depth',
    type=float, default=10.0, help='Maximum value of dataset depth')

# Spatial pyramid pool settings
parser.add_argument('--max_pool_sizes_spatial_pyramid_pool',
    nargs='+', type=int, default=[13, 17, 19, 21, 25], help='List of pool sizes for spatial pyramid pooling')
parser.add_argument('--n_convolution_spatial_pyramid_pool',
    type=int, default=3, help='Number of convolutions to use for spatial pyramid pooling')
parser.add_argument('--n_filter_spatial_pyramid_pool',
    type=int, default=8, help='Number of filters to use for spatial pyramid pooling')

# Network settings
parser.add_argument('--encoder_type',
    nargs='+', type=str, default=['vggnet08', 'spatial_pyramid_pool', 'batch_norm'], help='Encoder type')
parser.add_argument('--n_filters_encoder',
    nargs='+', type=int, default=[16, 32, 64, 128, 256], help='Number of filters to each in each encoder block')
parser.add_argument('--decoder_type',
    nargs='+', type=str, default=['multi-scale', 'uncertainty', 'batch_norm'], help='Decoder type')
parser.add_argument('--n_filters_decoder',
    nargs='+', type=int, default=[256, 128, 128, 64, 32], help='Number of filters to each in each decoder block')
parser.add_argument('--n_output_resolution',
    type=int, default=1, help='Number of output resolutions')
parser.add_argument('--min_predict_depth',
    type=float, default=0.1, help='Minimum value of depth prediction')
parser.add_argument('--max_predict_depth',
    type=float, default=10.0, help='Maximum value of depth prediction')

# Weight settings
parser.add_argument('--weight_initializer',
    type=str, default='xavier_normal', help='Weight initializers')
parser.add_argument('--activation_func',
    type=str, default='leaky_relu', help='Activation function')

# Training settings
parser.add_argument('--learning_rates',
    nargs='+', type=float, default=[1e-4, 5e-5], help='Space delimited list of learning rates')
parser.add_argument('--learning_schedule',
    nargs='+', type=int, default=[4, 10], help='Space delimited list to change learning rate')
parser.add_argument('--freeze_network_modules',
    nargs='+', type=str, default=None, help='List of modules to freeze in network')

# Augmentation settings
parser.add_argument('--augmentation_probabilities',
    nargs='+', type=float, default=[1.00], help='Probabilities to use data augmentation. Note: there is small chance that no augmentation take place even when we enter augmentation pipeline.')
parser.add_argument('--augmentation_schedule',
    nargs='+', type=int, default=[-1], help='If not -1, then space delimited list to change augmentation probability')
parser.add_argument('--augmentation_random_crop_type',
    nargs='+', type=str, default=['none'], help='Random crop type for data augmentation: horizontal, vertical, anchored, bottom, none')
parser.add_argument('--augmentation_random_crop_to_shape',
    nargs='+', type=int, default=[-1, -1], help='Shape after cropping')
parser.add_argument('--augmentation_random_flip_type',
    nargs='+', type=str, default=['none'], help='Random flip type for data augmentation: horizontal, vertical, none')
parser.add_argument('--augmentation_random_remove_points',
    nargs='+', type=float, default=[-1, -1], help='If set, randomly remove points from sparse depth')
parser.add_argument('--augmentation_random_noise_type',
    type=str, default='none', help='Random noise to add: gaussian, uniform')
parser.add_argument('--augmentation_random_noise_spread',
    type=float, default=-1, help='If gaussian noise, then standard deviation; if uniform, then min-max range')

# Loss function settings
parser.add_argument('--loss_func',
    nargs='+', type=str, default=['supervised_l1_normalized'], help='Loss functions available')
parser.add_argument('--w_supervised',
    type=float, default=1.00, help='Weight of supervised loss')
parser.add_argument('--w_weight_decay',
    type=float, default=0.00, help='Weight of weight decay regularization')

# Evaluation settings
parser.add_argument('--min_evaluate_depth',
    type=float, default=0.2, help='Minimum value of depth to evaluate')
parser.add_argument('--max_evaluate_depth',
    type=float, default=10.0, help='Maximum value of depth to evaluate')

# Checkpoint settings
parser.add_argument('--n_step_per_summary',
    type=int, default=1000, help='Number of iterations for logging summary')
parser.add_argument('--n_image_per_summary',
    type=int, default=4, help='Number of images to log in each summary')
parser.add_argument('--n_step_per_checkpoint',
    type=int, default=1000, help='Number of iterations for each checkpoint')
parser.add_argument('--checkpoint_path',
    type=str, default=None, help='Path to save checkpoints')
parser.add_argument('--start_step_validation',
    type=int, default=0, help='Number of steps before starting validation')
parser.add_argument('--restore_path',
    type=str, default=None, help='Path to restore model from checkpoint')

# Hardware settings
parser.add_argument('--device',
    type=str, default='cuda', help='Device to use: gpu, cpu')
parser.add_argument('--n_thread',
    type=int, default=8, help='Number of threads for fetching')


args = parser.parse_args()

if __name__ == '__main__':

    # Assert arguments
    assert len(args.learning_rates) == len(args.learning_schedule)

    args.encoder_type = [
        encoder_type.lower() for encoder_type in args.encoder_type
    ]

    assert len(args.n_filters_encoder) == 5

    args.decoder_type = [
        decoder_type.lower() for decoder_type in args.decoder_type
    ]

    assert len(args.n_filters_decoder) == 5

    args.weight_initializer = args.weight_initializer.lower()

    args.activation_func = args.activation_func.lower()

    args.loss_func = [
        loss_func.lower() for loss_func in args.loss_func
    ]

    args.cap_dataset_depth_method = args.cap_dataset_depth_method.lower()

    args.device = args.device.lower()

    if args.device not in ['gpu', 'cpu', 'cuda']:
        args.device = 'cuda'

    args.device = 'cuda' if args.device == 'gpu' else args.device

    train(train_sparse_depth_path=args.train_sparse_depth_path,
          train_ground_truth_path=args.train_ground_truth_path,
          val_sparse_depth_path=args.val_sparse_depth_path,
          val_ground_truth_path=args.val_ground_truth_path,
          # Batch settings
          n_batch=args.n_batch,
          n_height=args.n_height,
          n_width=args.n_width,
          # Dataset settings
          cap_dataset_depth_method=args.cap_dataset_depth_method,
          min_dataset_depth=args.min_dataset_depth,
          max_dataset_depth=args.max_dataset_depth,
          # Spatial pyramid pool settings
          max_pool_sizes_spatial_pyramid_pool=args.max_pool_sizes_spatial_pyramid_pool,
          n_convolution_spatial_pyramid_pool=args.n_convolution_spatial_pyramid_pool,
          n_filter_spatial_pyramid_pool=args.n_filter_spatial_pyramid_pool,
          # Network settings
          encoder_type=args.encoder_type,
          n_filters_encoder=args.n_filters_encoder,
          decoder_type=args.decoder_type,
          n_filters_decoder=args.n_filters_decoder,
          n_output_resolution=args.n_output_resolution,
          min_predict_depth=args.min_predict_depth,
          max_predict_depth=args.max_predict_depth,
          # Weight settings
          weight_initializer=args.weight_initializer,
          activation_func=args.activation_func,
          # Training settings
          learning_rates=args.learning_rates,
          learning_schedule=args.learning_schedule,
          freeze_network_modules=args.freeze_network_modules,
          # Augmentation settings
          augmentation_probabilities=args.augmentation_probabilities,
          augmentation_schedule=args.augmentation_schedule,
          augmentation_random_crop_type=args.augmentation_random_crop_type,
          augmentation_random_crop_to_shape=args.augmentation_random_crop_to_shape,
          augmentation_random_flip_type=args.augmentation_random_flip_type,
          augmentation_random_remove_points=args.augmentation_random_remove_points,
          augmentation_random_noise_type=args.augmentation_random_noise_type,
          augmentation_random_noise_spread=args.augmentation_random_noise_spread,
          # Loss function settings
          loss_func=args.loss_func,
          w_supervised=args.w_supervised,
          w_weight_decay=args.w_weight_decay,
          # Evaluation settings
          min_evaluate_depth=args.min_evaluate_depth,
          max_evaluate_depth=args.max_evaluate_depth,
          # Checkpoint settings
          n_step_per_summary=args.n_step_per_summary,
          n_image_per_summary=args.n_image_per_summary,
          n_step_per_checkpoint=args.n_step_per_checkpoint,
          checkpoint_path=args.checkpoint_path,
          start_step_validation=args.start_step_validation,
          restore_path=args.restore_path,
          # Hardware settings
          device=args.device,
          n_thread=args.n_thread)
