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
import tensorflow as tf
import tensorflow.contrib.slim as slim
import networks, loss_utils, losses, log_utils, net_utils
import global_constants as settings


class FusionNetModel(object):

    def __init__(self,
                 image0,
                 input_depth,
                 image1=None,
                 image2=None,
                 intrinsics=None,
                 ground_truth=None,
                 is_training=True,
                 # Network architecture
                 network_type=settings.NETWORK_TYPE_FUSIONNET,
                 image_filter_pct=settings.IMAGE_FILTER_PCT,
                 depth_filter_pct=settings.DEPTH_FILTER_PCT,
                 activation_func=settings.ACTIVATION_FUNC,
                 # Depth prediction settings
                 min_predict_depth=settings.MIN_PREDICT_DEPTH,
                 max_predict_depth=settings.MAX_PREDICT_DEPTH,
                 min_scale_depth=settings.MIN_SCALE_DEPTH,
                 max_scale_depth=settings.MAX_SCALE_DEPTH,
                 min_residual_depth=settings.MIN_RESIDUAL_DEPTH,
                 max_residual_depth=settings.MAX_RESIDUAL_DEPTH,
                 # Loss function
                 validity_map_color=settings.VALIDITY_MAP_COLOR,
                 w_color=settings.W_COLOR,
                 w_structure=settings.W_STRUCTURE,
                 w_sparse_depth=settings.W_SPARSE_DEPTH,
                 w_ground_truth=settings.W_GROUND_TRUTH,
                 w_smoothness=settings.W_SMOOTHNESS,
                 w_prior_depth=settings.W_PRIOR_DEPTH,
                 residual_threshold_prior_depth=settings.RESIDUAL_THRESHOLD_PRIOR_DEPTH,
                 rotation_param=settings.ROTATION_PARAM):

        # Input data
        self.image0 = image0
        self.image1 = image1
        self.image2 = image2
        self.intrinsics = intrinsics
        self.prior_depth = tf.expand_dims(input_depth[..., 0], axis=-1)
        self.ground_truth = ground_truth

        # Depth prediction range
        self.min_predict_depth = min_predict_depth
        self.max_predict_depth = max_predict_depth
        self.min_scale_depth = min_scale_depth
        self.max_scale_depth = max_scale_depth
        self.min_residual_depth = min_residual_depth
        self.max_residual_depth = max_residual_depth

        # Loss function coefficients
        self.validity_map_color = validity_map_color
        self.w_color = w_color
        self.w_structure = w_structure
        self.w_smoothness = w_smoothness
        self.w_sparse_depth = w_sparse_depth
        self.w_ground_truth = w_ground_truth
        self.w_prior_depth = w_prior_depth
        self.residual_threshold_prior_depth = residual_threshold_prior_depth

        # Data dimensions
        self.shape = self.image0.get_shape().as_list()

        # Extract sparse depth from input depth
        self.sparse_depth = \
            tf.expand_dims(input_depth[..., 1], axis=-1)

        # If non-zero then remove points with large discrepancy in neighborhood
        self.sparse_depth = net_utils.remove_outliers(
            self.sparse_depth,
            threshold=1.5,
            kernel_size=7)

        # Validity map is non-zero points in sparse depth
        self.validity_map_sparse_depth = tf.where(
            self.sparse_depth > 0,
            tf.ones_like(self.sparse_depth),
            tf.zeros_like(self.sparse_depth))

        # Scale the local region based on sparse depth
        local_scale = tf.where(
            self.sparse_depth > 0,
            self.sparse_depth / (self.prior_depth + 1e-6),
            self.sparse_depth)

        # If scale is very large then ignore it
        local_scale = tf.where(
            local_scale > 5,
            tf.ones_like(local_scale),
            local_scale)

        # Get scale for local neighborhood
        local_scale = slim.max_pool2d(
            local_scale,
            kernel_size=[5, 5],
            stride=1,
            padding='SAME')

        local_scale = tf.where(
            local_scale > 0,
            local_scale,
            tf.ones_like(local_scale))

        # Keep original sparse depth and scale the rest
        self.prior_depth = tf.where(
            self.validity_map_sparse_depth == 1,
            self.sparse_depth,
            self.prior_depth * local_scale)

        # Stack prior depth and sparse depth back together
        input_depth = tf.concat([
            self.prior_depth,
            self.sparse_depth],
            axis=-1)

        # Select activation function for network
        if activation_func == 'relu':
            activation_fn = tf.nn.relu
        elif activation_func == 'leaky_relu':
            activation_fn = tf.nn.leaky_relu
        elif activation_func == 'elu':
            activation_fn = tf.nn.elu
        else:
            raise ValueError('Unsupported activation function: {}'.format(activation_func))

        # Forward through network
        if network_type == 'fusionnet05':
            self.output_depth = networks.fusionnet05(
                image0,
                input_depth,
                activation_fn=activation_fn,
                image_filter_pct=image_filter_pct,
                depth_filter_pct=depth_filter_pct)[-1]
        elif network_type == 'fusionnet08':
            self.output_depth = networks.fusionnet08(
                image0,
                input_depth,
                activation_fn=activation_fn,
                image_filter_pct=image_filter_pct,
                depth_filter_pct=depth_filter_pct)[-1]
        else:
            raise ValueError('Unsupported architecture: {}'.format(network_type))

        # Split output depth into scale (alpha) and residual (beta)
        self.output_scale = \
            tf.expand_dims(self.output_depth[..., 0], axis=-1)
        self.output_residual = \
            tf.expand_dims(self.output_depth[..., 1], axis=-1)

        # Set scale between min and max scale depth
        self.output_scale = \
            (max_scale_depth - min_scale_depth) * self.output_scale + min_scale_depth

        # Set residual between min and max residual depth
        self.output_residual = tf.clip_by_value(
            self.output_residual,
            clip_value_min=min_residual_depth,
            clip_value_max=max_residual_depth)

        # Multiply by scale and add residual: \alpha(x) d(x) + \beta(x)
        self.output_depth = self.output_scale * self.prior_depth + self.output_residual

        # Prediction
        self.predict = self.output_depth

        if is_training:
            self.pose = networks.posenet(tf.concat([
                tf.concat([image0, image1], axis=-1),
                tf.concat([image0, image2], axis=-1)], axis=0),
                is_training=is_training)

            if rotation_param == 'euler':
                # Euler parametrization for rotation
                self.pose01, self.pose02 = [
                    loss_utils.pose_vec2mat(v) for v in tf.split(self.pose, 2, axis=0)
                ]
            elif rotation_param == 'exponential':
                # Exponential parametrization for rotation
                self.pose01, self.pose02 = [
                    loss_utils.pose_expm(v) for v in tf.split(self.pose, 2, axis=0)
                ]
            else:
                raise ValueError('Unsupport rotation parameterization: {}'.format(rotation_param))

            # Build loss function
            self.loss = self.build_loss()

    def build_loss(self):
        '''
        Temporal (video) rigid warping
        '''
        # Compute flow from image 0 to image 1
        flow01 = loss_utils.compute_rigid_flow(
            tf.squeeze(self.output_depth, axis=3),
            pose=self.pose01,
            intrinsics=self.intrinsics)

        # Compute flow from image 0 to image 2
        flow02 = loss_utils.compute_rigid_flow(
            tf.squeeze(self.output_depth, axis=3),
            pose=self.pose02,
            intrinsics=self.intrinsics)

        # Reconstruct im0 using im1 with rigid flow
        image01 = tf.reshape(loss_utils.flow_warp(self.image1, flow01), self.shape)

        # Reconstruct im0 using im2 with rigid flow
        image02 = tf.reshape(loss_utils.flow_warp(self.image2, flow02), self.shape)

        '''
        Construct loss function
        '''
        if self.validity_map_color == 'nonsparse':
            validity_map_color = 1.0 - self.validity_map_sparse_depth
        elif self.validity_map_color == 'all':
            validity_map_color = tf.ones_like(self.validity_map_sparse_depth)

        # Construct color consistency reconstruction loss
        loss_color01 = losses.color_consistency_loss_func(
            self.image0,
            image01,
            validity_map_color)
        loss_color02 = losses.color_consistency_loss_func(
            self.image0,
            image02,
            validity_map_color)
        loss_color = loss_color01 + loss_color02

        # Construct structural reconstruction loss
        loss_structure01 = losses.structural_loss_func(
            self.image0,
            image01,
            validity_map_color)
        loss_structure02 = losses.structural_loss_func(
            self.image0,
            image02,
            validity_map_color)
        loss_structure = loss_structure01 + loss_structure02

        # Construct sparse depth loss
        if self.w_sparse_depth > 0.0:
            loss_sparse_depth = losses.sparse_depth_loss_func(
                self.output_depth,
                self.sparse_depth,
                self.validity_map_sparse_depth)
        else:
            loss_sparse_depth = 0.0

        # Construct smoothness loss
        loss_smoothness = \
            losses.smoothness_loss_func(self.output_depth, self.image0)

        if self.w_prior_depth > 0.0:

            # Using residual to determine where to enforce prior
            if self.residual_threshold_prior_depth > 0.0:
                # Project using prior
                flow01_prior_depth = loss_utils.compute_rigid_flow(
                    tf.squeeze(self.prior_depth, axis=3),
                    pose=self.pose01,
                    intrinsics=self.intrinsics)
                image01_prior_depth = \
                    tf.reshape(loss_utils.flow_warp(self.image1, flow01_prior_depth), self.shape)

                # Compare residuals
                delta_image01_output_depth = tf.reduce_sum(
                    tf.abs(self.image0 - image01),
                    axis=-1,
                    keepdims=True)
                delta_image01_prior_depth = tf.reduce_sum(
                    tf.abs(self.image0 - image01_prior_depth),
                    axis=-1,
                    keepdims=True)

                # If global residual < threshold
                global_flag = tf.cond(
                    loss_color < self.residual_threshold_prior_depth,
                    lambda: 1.0,
                    lambda: 0.0)

                local_weights = tf.where(
                    delta_image01_output_depth > delta_image01_prior_depth,
                    tf.ones_like(self.prior_depth),
                    tf.zeros_like(self.prior_depth))

                w = global_flag * local_weights
            else:
                w = tf.ones_like(self.prior_depth)

            loss_prior_depth = losses.prior_depth_loss_func(
                self.output_depth,
                self.prior_depth,
                w)
        else:
            loss_prior_depth = 0.0

        if self.w_ground_truth > 0.0:
            # Create validity map on ground truth
            self.validity_map_ground_truth = tf.where(
                self.ground_truth > 0,
                tf.ones_like(self.ground_truth),
                tf.zeros_like(self.ground_truth))

            loss_ground_truth = losses.sparse_depth_loss_func(
                self.output_depth,
                self.ground_truth,
                self.validity_map_ground_truth)

        # Construct total loss
        loss = self.w_color * loss_color + \
            self.w_structure * loss_structure + \
            self.w_smoothness * loss_smoothness + \
            self.w_sparse_depth * loss_sparse_depth + \
            self.w_prior_depth * loss_prior_depth + \
            self.w_ground_truth * loss_ground_truth

        # Construct summary
        with tf.name_scope('fusionnet'):
            tf.summary.scalar('loss_color', loss_color)
            tf.summary.scalar('loss_structure', loss_structure)
            tf.summary.scalar('loss_smoothness', loss_smoothness)
            tf.summary.scalar('loss_sparse_depth', loss_sparse_depth)

            if self.w_prior_depth > 0.0:
                tf.summary.scalar('loss_prior_depth', loss_prior_depth)

            tf.summary.scalar('loss', loss)

            self.delta_depth = \
                (self.output_depth - self.prior_depth) / (self.prior_depth + 1e-6)

            # Log histogram
            tf.summary.histogram('output_depth_distro', self.output_depth)
            tf.summary.histogram('output_scale_distro', self.output_scale)
            tf.summary.histogram('output_residual_distro', self.output_residual)
            tf.summary.histogram('prior_depth_distro', self.prior_depth)
            tf.summary.histogram('delta_depth_distro', self.delta_depth)

            # Visualize reconstruction
            tf.summary.image(
                'image0_image01_image02',
                tf.concat([self.image0, image01, image02], axis=1),
                max_outputs=3)

            # Visualize depth maps
            tf.summary.image(
                'image0_output_prior_delta',
                tf.concat([
                    self.image0,
                    log_utils.gray2color(
                        self.output_depth,
                        'viridis',
                        vmin=self.min_predict_depth,
                        vmax=self.max_predict_depth),
                    log_utils.gray2color(
                        self.prior_depth,
                        'viridis',
                        vmin=self.min_predict_depth,
                        vmax=self.max_predict_depth),
                    log_utils.gray2color(
                        self.delta_depth,
                        'cividis',
                        vmin=0.80,
                        vmax=1.20)], axis=1),
                max_outputs=3)

        with tf.name_scope('posenet'):
            tf.summary.histogram('tx01_distro', self.pose01[:, 0, 3])
            tf.summary.histogram('ty01_distro', self.pose01[:, 1, 3])
            tf.summary.histogram('tz01_distro', self.pose01[:, 2, 3])
            tf.summary.histogram('tx02_distro', self.pose02[:, 0, 3])
            tf.summary.histogram('ty02_distro', self.pose02[:, 1, 3])
            tf.summary.histogram('tz02_distro', self.pose02[:, 2, 3])

        # Return total loss
        return loss
