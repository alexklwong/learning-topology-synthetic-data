'''
Network parameters
'''
# Input image dimensions
N_BATCH                             = 8
N_HEIGHT                            = 320
N_WIDTH                             = 768
N_CHANNEL                           = 3

# Dataloader settings
DEPTH_LOAD_MULTIPLIER               = 256.0
MIN_DATASET_DEPTH                   = 0.00
MAX_DATASET_DEPTH                   = 655.00

# Preprocessing
OUTLIER_REMOVAL_THRESHOLD           = 1.5
OUTLIER_REMOVAL_KERNEL_SIZE         = 7

# Network architecture
NETWORK_TYPE_SCAFFNET               = 'scaffnet32'
NETWORK_TYPE_FUSIONNET              = 'vgg11'
ACTIVATION_FUNC                     = 'leaky_relu'
OUTPUT_FUNC                         = 'linear'
IM_FILTER_PCT                       = 0.75
ZV_FILTER_PCT                       = 0.25
N_FILTER_OUTPUT                     = 0

# Spatial pyramid pooling
POOL_KERNEL_SIZES_SPP               = [5, 7, 9, 11]
N_CONVOLUTION_SPP                   = 3

# Depth prediction settings
MIN_PREDICT_DEPTH                   = 1.50
MAX_PREDICT_DEPTH                   = 100.00
MIN_SCALE_DEPTH                     = 0.25
MAX_SCALE_DEPTH                     = 4.00
MIN_RESIDUAL_DEPTH                  = -1000.0
MAX_RESIDUAL_DEPTH                  = 1000.0

# Training settings
N_EPOCH                             = 30
LEARNING_RATES                      = [1.0e-4, 5.0e-5, 2.5e-5]
LEARNING_SCHEDULE                   = [0.60, 0.80]
LOSS_FUNC_SCAFFNET                  = 'l1_norm'

# Loss function
W_SUPERVISED                        = 1.00

W_PH                                = 1.00
W_CO                                = 0.20
W_ST                                = 0.80
W_SM                                = 0.01
W_SZ                                = 0.20
W_PZ                                = 0.10
ROT_PARAM                           = 'euler'

# Depth evaluation settings
MIN_EVALUATE_DEPTH                  = 0.0
MAX_EVALUATE_DEPTH                  = 100.00

'''
Model checkpoints
'''
N_CHECKPOINT                        = 5000
N_SUMMARY                           = 500
CHECKPOINT_PATH                     = 'log'
RESTORE_PATH                        = ''

'''
Hardware settings
'''
N_THREAD                            = 8
