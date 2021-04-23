import tensorflow as tf
import networks, log_utils, losses
import global_constants as settings


class ScaffNetModel(object):
    '''
    ScaffNet model class for learning dense topology from sparse points

    Args:
        input_depth : tensor
            N x H x W x 1 input sparse depth
        ground_truth : tensor
            N x H x W x 1 ground truth
        is_training : bool
            if set, then builds loss function
        network_type : str
            network type to build
        activation_func : str
            activation function to use in network
        output_func : str
            output function to use
        n_filter_output : int
            number of filters to use in predicting full resolution output
        pool_kernel_sizes_spp : list[int]
            kernel sizes to use for spatial pyramid pooling
        n_convolution_spp : int
            number of 1 x 1 convolutions to balance density vs. details trade-off
        n_filter_spp : int
            number of filters to use in 1 x 1 convolutions in spatial pyramid pooling
        min_dataset_depth : float
            minimum depth value to consider, if less than actual minimium of dataset then cap
        max_dataset_depth : float
            maximum depth value to consider, if more than actual maximium of dataset then cap
        min_predict_depth : float
            minimum depth value to predict
        max_predict_depth : float
            maximum depth value to predict
        loss_func : str
            loss function to minimize
        w_supervised : float
            weight of supervised loss
    '''
    def __init__(self,
                 input_depth,
                 ground_truth,
                 is_training=True,
                 network_type=settings.NETWORK_TYPE_SCAFFNET,
                 activation_func=settings.ACTIVATION_FUNC,
                 output_func=settings.OUTPUT_FUNC,
                 n_filter_output=settings.N_FILTER_OUTPUT,
                 pool_kernel_sizes_spp=settings.POOL_KERNEL_SIZES_SPP,
                 n_convolution_spp=settings.N_CONVOLUTION_SPP,
                 n_filter_spp=settings.N_FILTER_SPP,
                 min_dataset_depth=settings.MIN_DATASET_DEPTH,
                 max_dataset_depth=settings.MAX_DATASET_DEPTH,
                 min_predict_depth=settings.MIN_PREDICT_DEPTH,
                 max_predict_depth=settings.MAX_PREDICT_DEPTH,
                 loss_func=settings.LOSS_FUNC_SCAFFNET,
                 w_supervised=settings.W_SUPERVISED):

        self.shape = input_depth.get_shape().as_list()
        self.min_dataset_depth = min_dataset_depth
        self.max_dataset_depth = max_dataset_depth
        self.min_predict_depth = min_predict_depth
        self.max_predict_depth = max_predict_depth
        self.loss_func = loss_func
        self.w_supervised = w_supervised

        # Perform min-max cap on ground-truth
        ground_truth_capped = tf.clip_by_value(
            ground_truth[..., 0],
            self.min_dataset_depth,
            self.max_dataset_depth)
        ground_truth_capped = tf.expand_dims(ground_truth_capped, axis=-1)

        validity_map_ground_truth = tf.where(
            ground_truth[..., 0] > 0,
            tf.ones(self.shape[0:3]),
            tf.zeros(self.shape[0:3]))

        self.validity_map_ground_truth = tf.expand_dims(
            validity_map_ground_truth * ground_truth[..., 1],
            axis=-1)

        self.ground_truth = \
            self.validity_map_ground_truth * ground_truth_capped

        # Perform min-max cap on sparse input
        sparse_depth_capped = tf.clip_by_value(
            input_depth[..., 0],
            self.min_dataset_depth,
            self.max_dataset_depth)
        sparse_depth_capped = tf.expand_dims(sparse_depth_capped, axis=-1)

        validity_map_sparse_depth = tf.where(
            input_depth[..., 0] > 0,
            tf.ones(self.shape[0:3]),
            tf.zeros(self.shape[0:3]))

        self.validity_map_sparse_depth = tf.expand_dims(
            validity_map_sparse_depth * input_depth[..., 1],
            axis=-1)

        self.sparse_depth = \
            self.validity_map_sparse_depth * sparse_depth_capped

        input_depth = tf.concat(
            [self.sparse_depth, self.validity_map_sparse_depth],
            axis=-1)

        # Select activation function for network
        if activation_func == 'relu':
            activation_fn = tf.nn.relu
        elif activation_func == 'leaky_relu':
            activation_fn = tf.nn.leaky_relu
        elif activation_func == 'elu':
            activation_fn = tf.nn.elu
        else:
            raise ValueError('Invalid activation function: {}'.format(activation_func))

        # Select output function for network
        if output_func == 'identity' or output_func == 'linear':
            output_fn = tf.identity
        elif output_func == 'sigmoid':
            output_fn = tf.nn.sigmoid
        else:
            raise ValueError('Invalid output function: {}'.format(output_func))

        # Forward through network
        if network_type == 'scaffnet16':
            self.output_depth = networks.scaffnet16(
                input_depth,
                n_output=1,
                activation_fn=activation_fn,
                output_fn=output_fn,
                n_filter_output=n_filter_output,
                pool_kernel_sizes_spp=pool_kernel_sizes_spp,
                n_convolution_spp=n_convolution_spp,
                n_filter_spp=n_filter_spp)
        elif network_type == 'scaffnet32':
            self.output_depth = networks.scaffnet32(
                input_depth,
                n_output=1,
                activation_fn=activation_fn,
                output_fn=output_fn,
                n_filter_output=n_filter_output,
                pool_kernel_sizes_spp=pool_kernel_sizes_spp,
                n_convolution_spp=n_convolution_spp,
                n_filter_spp=n_filter_spp)

        self.output_depth = self.output_depth[-1]

        if output_func == 'sigmoid':
            self.output_depth = \
                self.min_predict_depth / (self.output_depth + self.min_predict_depth / self.max_predict_depth)

        # Prediction
        self.predict = self.output_depth

        if is_training:
            # Build loss function
            self.loss = self.build_loss()

    def build_loss(self):

        loss = losses.l1_loss_func(
            src=self.output_depth,
            tgt=self.ground_truth,
            v=self.validity_map_ground_truth,
            normalize=True if 'norm' in self.loss_func else False)

        loss = self.w_supervised * loss

        # Construct summary
        with tf.name_scope('scaffnet'):
            tf.summary.scalar('loss', loss)
            tf.summary.histogram('sparse_depth_distro', self.sparse_depth)
            tf.summary.histogram('ground_truth_distro', self.ground_truth)
            tf.summary.histogram('output_depth_distro', self.output_depth)

            loss_sparse_depth = losses.l1_loss_func(
                src=self.predict,
                tgt=self.sparse_depth,
                v=self.validity_map_sparse_depth,
                normalize=True)
            tf.summary.scalar('loss_sparse_depth', loss_sparse_depth)

            # Visualize depth maps
            tf.summary.image('sparse_depth-ground_truth-output_depth-error',
                tf.concat([
                    log_utils.gray2color(self.sparse_depth, colormap='viridis'),
                    log_utils.gray2color(self.ground_truth, colormap='viridis'),
                    log_utils.gray2color(self.predict, colormap='viridis'),
                    log_utils.gray2color(
                        tf.where(self.validity_map_ground_truth > 0,
                            tf.abs(self.predict - self.ground_truth) / self.ground_truth,
                            self.validity_map_ground_truth),
                        colormap='magma',
                        vmin=0.0,
                        vmax=0.20)], axis=1),
                max_outputs=3)

        # Return total loss
        return loss
