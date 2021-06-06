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
# Input image dimensions
N_BATCH                             = 8
N_HEIGHT                            = 320
N_WIDTH                             = 768
N_CHANNEL                           = 3

# Dataloader settings
CROP_TYPE                           = 'bottom'
MIN_DATASET_DEPTH                   = 0.00
MAX_DATASET_DEPTH                   = 655.00

# Preprocessing
OUTLIER_REMOVAL_THRESHOLD           = 1.5
OUTLIER_REMOVAL_KERNEL_SIZE         = 7

# General network architecture settings
ACTIVATION_FUNC                     = 'leaky_relu'

# ScaffNet settings
NETWORK_TYPE_SCAFFNET               = 'scaffnet32'
N_FILTER_OUTPUT_SCAFFNET            = 0

# Spatial pyramid pooling
POOL_KERNEL_SIZES_SPP               = [5, 7, 9, 11]
N_CONVOLUTION_SPP                   = 3
N_FILTER_SPP                        = 32

# ScaffNet depth prediction settings
MIN_PREDICT_DEPTH                   = 1.50
MAX_PREDICT_DEPTH                   = 100.00

# ScaffNet loss function
W_SUPERVISED                        = 1.00

# FusionNet settings
NETWORK_TYPE_FUSIONNET              = 'vggnet08'
IMAGE_FILTER_PCT                    = 0.75
DEPTH_FILTER_PCT                    = 0.25

# FusionNet depth prediction settings
MIN_SCALE_DEPTH                     = 0.25
MAX_SCALE_DEPTH                     = 4.00
MIN_RESIDUAL_DEPTH                  = -1000.0
MAX_RESIDUAL_DEPTH                  = 1000.0

# Training settings
N_EPOCH                             = 30
LEARNING_RATES                      = [1.0e-4, 5.0e-5, 2.5e-5]
LEARNING_SCHEDULE                   = [0.60, 0.80]
LOSS_FUNC_SCAFFNET                  = 'l1_norm'

# FusionNet loss function
VALIDITY_MAP_COLOR                  = 'nonsparse'
W_COLOR                             = 0.20
W_STRUCTURE                         = 0.80
W_SMOOTHNESS                        = 0.01
W_SPARSE_DEPTH                      = 0.20
W_GROUND_TRUTH                      = 0.00
W_PRIOR_DEPTH                       = 0.10
RESIDUAL_THRESHOLD_PRIOR_DEPTH      = 0.30
ROTATION_PARAM                      = 'euler'

# Depth evaluation settings
MIN_EVALUATE_DEPTH                  = 0.0
MAX_EVALUATE_DEPTH                  = 100.00

# Checkpoint settings
N_CHECKPOINT                        = 5000
N_SUMMARY                           = 500
CHECKPOINT_PATH                     = 'log'
RESTORE_PATH                        = ''

# Hardware settings
N_THREAD                            = 8

# Miscellaneous
EPSILON                             = 1e-10
