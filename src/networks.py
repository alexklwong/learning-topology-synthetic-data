import tensorflow as tf
import tensorflow.contrib.slim as slim


'''
Util for creating network layers and blocks
'''
def vgg2d(x,
          kernel_size,
          n_convolution=2,
          stride=2,
          padding='SAME',
          activation_fn=tf.nn.relu,
          use_pooling=False,
          reuse=tf.AUTO_REUSE,
          name=None):
    '''
    Creates a VGG block of n_conv layers

    Args:
        x : tensor
            input tensor
        kernel_size : list[int]
            3 x 1 list [k, k, f] of kernel size k, number of filters f
        n_convolution : int
            number of total convolutions
        stride : int
            stride size for downsampling
        padding : str
            padding on edges in case size doesn't match
        activation_fn : func
            activation function after convolution
        use_pooling : bool
            if set, use max pooling instead of the striding
        reuse : bool
            if set, reuse weights if have already been defined in same variable scope
        name : str
            name of node in computational graph
    Returns:
        tensor : layer after VGG convolutions
    '''

    name = name + '_vgg2d_' if name is not None else 'vgg2d_'
    layers = [x]

    for n in range(n_convolution - 1):
        layer_name = name + 'conv' + str(n + 1)

        conv = slim.conv2d(
            layers[-1],
            num_outputs=kernel_size[2],
            kernel_size=kernel_size[0:2],
            stride=1,
            padding=padding,
            activation_fn=activation_fn,
            reuse=reuse,
            scope=layer_name)

        layers.append(conv)

    layer_name = name + 'conv' + str(n_convolution)

    if use_pooling and stride > 1:
        convn = slim.conv2d(
            layers[-1],
            num_outputs=kernel_size[2],
            kernel_size=kernel_size[0:2],
            stride=1,
            padding=padding,
            activation_fn=activation_fn,
            reuse=reuse,
            scope=layer_name)

        convn = slim.max_pool2d(convn, kernel_size=[2, 2], padding=padding)
    else:
        convn = slim.conv2d(
            layers[-1],
            num_outputs=kernel_size[2],
            kernel_size=kernel_size[0:2],
            stride=stride,
            padding=padding,
            activation_fn=activation_fn,
            reuse=reuse,
            scope=layer_name)

    return convn

def upconv2d(x,
             shape,
             kernel_size,
             stride=2,
             padding='SAME',
             activation_fn=tf.nn.relu,
             reuse=tf.AUTO_REUSE,
             name=None):
    '''
    Creates a 2D up-convolution layer upsample and convolution

    Args:
        x : tensor
            input tensor
        shape : list[int]
            2 element list of tensor y-x shape
        kernel_size : list[int]
            3 x 1 list [k, k, f] of kernel size k, number of filters f
        stride : int
            stride size of convolution
        padding : str
            padding on edges in case size doesn't match
        activation_fn : func
            activation function after convolution
        reuse : bool
            if set, reuse weights if have already been defined in same variable scope
        name : str
            name of node in computational graph
    Returns:
        tensor : layer after performing up-convolution
    '''

    layer_name = name if name is not None else 'upconv'

    x_up = tf.image.resize_nearest_neighbor(x, shape)

    conv = slim.conv2d(
        x_up,
        num_outputs=kernel_size[2],
        kernel_size=kernel_size[0:2],
        stride=stride,
        padding=padding,
        activation_fn=activation_fn,
        reuse=reuse,
        scope=layer_name)

    return conv

def spp2d(x,
          n_output,
          n_convolution=1,
          pool_kernel_sizes=[2, 5],
          padding='SAME',
          activation_fn=tf.nn.relu,
          reuse=tf.AUTO_REUSE,
          name=None):
    '''
    Creates spatial pyramid pooling with pooling rates and convolves the pyramid

    Args:
        x : tensor
            input tensor
        n_output : int
            number of output filters
        n_convolution : int
            number of 1 x 1 convolutions
        pool_kernel_sizes : list[int]
            kernel sizes for max pooling
        padding : str
            padding on edges in case size doesn't match
        activation_fn : func
            activation function after convolution
        reuse : bool
            if set, reuse weights if have already been defined in same variable scope
        name : str
            name of node in computational graph
    Returns:
        tensor : layer after spatial pyramid pooling
    '''

    name = name + '_spp2d_' if name is not None else 'spp2d_'
    layers = [x]

    # Perform multi-scale pooling
    for s in pool_kernel_sizes:
        layer = slim.max_pool2d(x, kernel_size=[s, s], stride=1, padding=padding)
        layers.append(layer)

    # Stack pooling layers together
    layers.append(tf.concat(layers, axis=-1))

    # Balance density vs. detail with 1 x 1 convolutions
    for n in range(n_convolution):
        layer_name = name + 'conv' + str(n + 1)

        conv = slim.conv2d(
            layers[-1],
            num_outputs=n_output,
            kernel_size=[1, 1],
            stride=1,
            padding=padding,
            activation_fn=activation_fn,
            reuse=reuse,
            scope=layer_name)

        layers.append(conv)

    return layers[-1]


'''
Network architectures
'''
def posenet(data, is_training):
    '''
    Creates a pose network that predicts 6 degrees of freedom pose

    Args:
        data : tensor
            input data N x H x W x D
        is_training : bool
            if set then network is training (matters on using batch norm, but is
            better to be explicit)
    Returns:
        tensor : 6 degrees of freedom pose
    '''

    batch_norm_params = { 'is_training': is_training }

    with tf.variable_scope('posenet'):
        with slim.arg_scope([slim.conv2d],
                            normalizer_fn=slim.batch_norm,
                            normalizer_params=batch_norm_params,
                            weights_regularizer=slim.l2_regularizer(0.0001),
                            activation_fn=tf.nn.relu):

            conv1 = slim.conv2d(data,  16,  7, 2)
            conv2 = slim.conv2d(conv1, 32,  5, 2)
            conv3 = slim.conv2d(conv2, 64,  3, 2)
            conv4 = slim.conv2d(conv3, 128, 3, 2)
            conv5 = slim.conv2d(conv4, 256, 3, 2)
            conv6 = slim.conv2d(conv5, 256, 3, 2)
            conv7 = slim.conv2d(conv6, 256, 3, 2)
            pose = slim.conv2d(conv7, 6, 1, 1, normalizer_fn=None, activation_fn=None)

            pose_mean = tf.reduce_mean(pose, [1, 2])

            return 0.01 * tf.reshape(pose_mean, [-1, 6])

def scaffnet16(depth,
               activation_fn=tf.nn.relu,
               output_fn=tf.identity,
               n_filter_output=0,
               pool_kernel_sizes_spp=[5, 7, 9, 11],
               n_convolution_spp=1,
               n_filter_spp=16):
    '''
    Creates ScaffNet with 16 initial filters

    Args:
        depth : tensor
            input sparse depth with validity map (N x H x W x 2)
        activation_fn : func
            activation function after convolution
        output_fn : func
            activation function to produce output predictions
        n_filter_output: int
            number of filters in the output layer if 0 then use upsample
        pool_kernel_size_spp : list[int]
            sets the max pooling rates for spatial pyramid pooling
        n_convolution_spp : int
            number of 1 x 1 convolutions in SPP
        n_filter_spp : int
            number of filters to use for 1 x 1 convolutions in SPP
    Returns:
        list[tensor] : list containing prediction and upsampled prediction at original resolution
    '''

    with tf.variable_scope('scaffnet16'):
        n_filters = [16, 32, 64, 96, 128]
        shape = depth.get_shape().as_list()[1:3]

        layers, skips = scaffnet_encoder(
            depth,
            n_mod1=1,
            n_mod2=1,
            n_mod3=1,
            n_mod4=1,
            n_mod5=1,
            n_filters=n_filters,
            activation_fn=activation_fn,
            pool_kernel_sizes_spp=pool_kernel_sizes_spp,
            n_convolution_spp=n_convolution_spp,
            n_filter_spp=n_filter_spp)

        layers, outputs = scaffnet_decoder(
            layers,
            skips,
            shape,
            n_filters=n_filters,
            activation_fn=activation_fn,
            output_fn=output_fn,
            n_filter_output=n_filter_output)

        return outputs

def scaffnet32(depth,
               activation_fn=tf.nn.relu,
               output_fn=tf.identity,
               n_filter_output=0,
               pool_kernel_sizes_spp=[5, 7, 9, 11],
               n_convolution_spp=1,
               n_filter_spp=32):
    '''
    Creates ScaffNet with 32 initial filters

    Args:
        depth : tensor
            input sparse depth with validity map (N x H x W x 2)
        activation_fn : func
            activation function after convolution
        output_fn : func
            activation function to produce output predictions
        n_filter_output: int
            number of filters in the output layer if 0 then use upsample
        pool_kernel_size_spp : list[int]
            sets the max pooling rates for spatial pyramid pooling
        n_convolution_spp : int
            number of 1 x 1 convolutions in SPP
        n_filter_spp : int
            number of filters to use for 1 x 1 convolutions in SPP
    Returns:
        list[tensor] : list containing prediction and upsampled prediction at original resolution
    '''

    with tf.variable_scope('scaffnet32'):
        n_filters = [32, 64, 96, 128, 196]
        shape = depth.get_shape().as_list()[1:3]

        layers, skips = scaffnet_encoder(
            depth,
            n_mod1=1,
            n_mod2=1,
            n_mod3=1,
            n_mod4=1,
            n_mod5=1,
            n_filters=n_filters,
            activation_fn=activation_fn,
            pool_kernel_sizes_spp=pool_kernel_sizes_spp,
            n_convolution_spp=n_convolution_spp,
            n_filter_spp=n_filter_spp)

        layers, outputs = scaffnet_decoder(
            layers,
            skips,
            shape,
            n_filters=n_filters,
            activation_fn=activation_fn,
            output_fn=output_fn,
            n_filter_output=n_filter_output)

        return outputs

def fusionnet05(image,
                depth,
                activation_fn=tf.nn.relu,
                output_fn=tf.identity,
                image_filter_pct=0.75,
                depth_filter_pct=0.25):
    '''
    Creates a FusionNet with 5 layers in each encoder branch

    Args:
        image : tensor
            input image (N x H x W x D)
        depth : tensor
            input sdepth with validity map (N x H x W x 2)
        activation_fn : func
            activation function after convolution
        output_fn : func
            activation function to produce output predictions
        image_filter_pct : float
            percent of parameters to allocate to the image branch
        depth_filter_pct : float
            percent of parameters to allocate to the depth branch
    Returns:
        list[tensor] : list containing prediction and upsampled prediction at original resolution
    '''

    with tf.variable_scope('fusionnet05'):
        shape = image.get_shape().as_list()[1:3]

        layers, skips = fusionnet_encoder(
            image,
            depth,
            n_mod1=1,
            n_mod2=1,
            n_mod3=1,
            n_mod4=1,
            n_mod5=1,
            activation_fn=activation_fn,
            image_filter_pct=image_filter_pct,
            depth_filter_pct=depth_filter_pct)

        layers, outputs = fusionnet_decoder(
            layers,
            skips,
            shape,
            activation_fn=activation_fn,
            output_fn=output_fn)

        return outputs

def fusionnet08(image,
                depth,
                activation_fn=tf.nn.relu,
                output_fn=tf.identity,
                image_filter_pct=0.75,
                depth_filter_pct=0.25):
    '''
    Creates a FusionNet with 8 layers in each encoder branch

    Args:
        image : tensor
            input image (N x H x W x D)
        depth : tensor
            input sparse depth with validity map (N x H x W x 2)
        activation_fn : func
            activation function after convolution
        output_fn : func
            activation function to produce output predictions
        image_filter_pct : float
            percent of parameters to allocate to the image branch
        depth_filter_pct : float
            percent of parameters to allocate to the depth branch
    Returns:
        list[tensor] : list containing prediction and upsampled prediction at original resolution
    '''

    with tf.variable_scope('fusionnet08'):
        shape = image.get_shape().as_list()[1:3]

        layers, skips = fusionnet_encoder(
            image,
            depth,
            n_mod1=1,
            n_mod2=1,
            n_mod3=2,
            n_mod4=2,
            n_mod5=2,
            activation_fn=activation_fn,
            image_filter_pct=image_filter_pct,
            depth_filter_pct=depth_filter_pct)

        layers, outputs = fusionnet_decoder(
            layers,
            skips,
            shape,
            activation_fn=activation_fn,
            output_fn=output_fn)

        return outputs


'''
Encoder architectures
'''
def fusionnet_encoder(image,
                      depth,
                      n_mod1=1,
                      n_mod2=2,
                      n_mod3=2,
                      n_mod4=2,
                      n_mod5=2,
                      image_filter_pct=0.75,
                      depth_filter_pct=0.25,
                      activation_fn=tf.nn.relu,
                      reuse_vars=tf.AUTO_REUSE):
    '''
    Creates a two branch encoder (one for image and the other for depth)
    with resolution from 1 to 1/32

    Args:
        image : tensor
            input image (N x H x W x D)
        depth : tensor
            input sparse depth with validity map (N x H x W x 2)
        n_mod<n> : int
            number of convolutional layers to perform in nth VGG block
        image_filter_pct : float
            percent of parameters to allocate to the image branch
        depth_filter_pct : float
            percent of parameters to allocate to the depth branch
        activation_fn : func
            activation function after convolution
        reuse_vars : bool
            if set, reuse weights if have already been defined in same variable scope
    Returns:
        list[tensor] : list containing all the layers last element is the
            latent representation (1/32 resolution)
        list[tensor] : list containing all the skip connections
    '''

    layers = []
    skips  = []
    padding = 'SAME'

    with tf.variable_scope('enc1', reuse=reuse_vars):
        kernel_size = [5, 5, 64]

        enc1_conv_image = vgg2d(
            image,
            kernel_size=kernel_size[0:2] + [int(image_filter_pct * kernel_size[2])],
            n_convolution=n_mod1,
            padding=padding,
            activation_fn=activation_fn,
            reuse=reuse_vars,
            name='image')

        enc1_conv_depth = vgg2d(
            depth,
            kernel_size=kernel_size[0:2] + [int(depth_filter_pct * kernel_size[2])],
            n_convolution=n_mod1,
            padding=padding,
            activation_fn=activation_fn,
            reuse=reuse_vars,
            name='depth')

        layers.append(enc1_conv_depth)
        layers.append(enc1_conv_image)
        skips.append(tf.concat([enc1_conv_depth, enc1_conv_image], axis=-1))

    with tf.variable_scope('enc2', reuse=reuse_vars):
        kernel_size = [3, 3, 128]

        enc2_conv_image = vgg2d(
            layers[-1],
            kernel_size=kernel_size[0:2] + [int(image_filter_pct * kernel_size[2])],
            n_convolution=n_mod2,
            padding=padding,
            activation_fn=activation_fn,
            reuse=reuse_vars,
            name='image')

        enc2_conv_depth = vgg2d(
            layers[-2],
            kernel_size=kernel_size[0:2] + [int(depth_filter_pct * kernel_size[2])],
            n_convolution=n_mod2,
            padding=padding,
            activation_fn=activation_fn,
            reuse=reuse_vars,
            name='depth')

        layers.append(enc2_conv_depth)
        layers.append(enc2_conv_image)
        skips.append(tf.concat([enc2_conv_depth, enc2_conv_image], axis=-1))

    with tf.variable_scope('enc3', reuse=reuse_vars):
        kernel_size = [3, 3, 256]

        enc3_conv_image = vgg2d(
            layers[-1],
            kernel_size=kernel_size[0:2] + [int(image_filter_pct * kernel_size[2])],
            n_convolution=n_mod3,
            padding=padding,
            activation_fn=activation_fn,
            reuse=reuse_vars,
            name='image')

        enc3_conv_depth = vgg2d(
            layers[-2],
            kernel_size=kernel_size[0:2] + [int(depth_filter_pct * kernel_size[2])],
            n_convolution=n_mod3,
            padding=padding,
            activation_fn=activation_fn,
            reuse=reuse_vars,
            name='depth')

        layers.append(enc3_conv_depth)
        layers.append(enc3_conv_image)
        skips.append(tf.concat([enc3_conv_depth, enc3_conv_image], axis=-1))

    with tf.variable_scope('enc4', reuse=reuse_vars):
        kernel_size = [3, 3, 512]

        enc4_conv_image = vgg2d(
            layers[-1],
            kernel_size=kernel_size[0:2] + [int(image_filter_pct * kernel_size[2])],
            n_convolution=n_mod4,
            padding=padding,
            activation_fn=activation_fn,
            reuse=reuse_vars,
            name='image')

        enc4_conv_depth = vgg2d(
            layers[-2],
            kernel_size=kernel_size[0:2] + [int(depth_filter_pct * kernel_size[2])],
            n_convolution=n_mod4,
            padding=padding,
            activation_fn=activation_fn,
            reuse=reuse_vars,
            name='depth')

        layers.append(enc4_conv_depth)
        layers.append(enc4_conv_image)
        skips.append(tf.concat([enc4_conv_depth, enc4_conv_image], axis=-1))

    with tf.variable_scope('enc5', reuse=reuse_vars):
        kernel_size = [3, 3, 512]

        enc5_conv_image = vgg2d(
            layers[-1],
            kernel_size=kernel_size[0:2] + [int(image_filter_pct * kernel_size[2])],
            n_convolution=n_mod5,
            padding=padding,
            activation_fn=activation_fn,
            reuse=reuse_vars,
            name='image')

        enc5_conv_depth = vgg2d(
            layers[-2],
            kernel_size=kernel_size[0:2] + [int(depth_filter_pct * kernel_size[2])],
            n_convolution=n_mod5,
            padding=padding,
            activation_fn=activation_fn,
            reuse=reuse_vars,
            name='depth')

        layers.append(tf.concat([enc5_conv_depth, enc5_conv_image], axis=-1))

    return layers, skips

def scaffnet_encoder(depth,
                     n_mod1=1,
                     n_mod2=1,
                     n_mod3=1,
                     n_mod4=1,
                     n_mod5=1,
                     n_filters=[32, 64, 96, 128, 196],
                     activation_fn=tf.nn.relu,
                     pool_kernel_sizes_spp=[5, 7, 9, 11],
                     n_convolution_spp=3,
                     n_filter_spp=32,
                     reuse_vars=tf.AUTO_REUSE):
    '''
    Creates Scaffnet encoder with resolution from 1 to 1/32

    Args:
        depth : tensor
            input sparse depth with validity map (N x H x W x 2)
        n_mod<n> : int
            number of convolutional layers to perform in nth VGG block
        n_filters : list[int]
            filters for each module
        activation_fn : func
            activation function after convolution
        pool_kernel_sizes_spp : list[int]
            sets the max pooling rates for spatial pyramid pooling
        n_convolution_spp : int
            number of 1 x 1 convolutions in SPP
        n_filter_spp : int
            number of filters to use for 1 x 1 convolutions in SPP
        reuse_vars : bool
            if set, reuse weights if have already been defined in same variable scope
    Returns:
        list[tensor] : list containing all the layers last element is the
            latent representation (1/32 resolution)
        list[tensor] : list containing all the skip connections
    '''
    layers = []
    skips  = []
    padding = 'SAME'
    with tf.variable_scope('enc1', reuse=reuse_vars):
        kernel_size = [5, 5, n_filters[0]]

        if len(pool_kernel_sizes_spp) > 0:
            spatial_pyramid_pooling = spp2d(
                depth,
                n_output=n_filter_spp,
                n_convolution=n_convolution_spp,
                pool_kernel_sizes=pool_kernel_sizes_spp,
                padding=padding,
                activation_fn=activation_fn,
                reuse=reuse_vars)

            layers.append(spatial_pyramid_pooling)

            enc1_conv = vgg2d(
                layers[-1],
                kernel_size=kernel_size,
                n_convolution=1,
                stride=2,
                padding=padding,
                activation_fn=activation_fn,
                use_pooling=False,
                reuse=reuse_vars)

            layers.append(enc1_conv)
        else:
            enc1_conv = vgg2d(
                depth,
                kernel_size=kernel_size,
                n_convolution=n_mod1,
                padding=padding,
                activation_fn=activation_fn,
                use_pooling=False,
                reuse=reuse_vars)

            layers.append(enc1_conv)

        skips.append(layers[-1])

    with tf.variable_scope('enc2', reuse=reuse_vars):
        kernel_size = [3, 3, n_filters[1]]

        enc2_conv = vgg2d(
            layers[-1],
            kernel_size=kernel_size,
            n_convolution=n_mod2,
            padding=padding,
            activation_fn=activation_fn,
            use_pooling=False,
            reuse=reuse_vars)

        layers.append(enc2_conv)
        skips.append(layers[-1])

    with tf.variable_scope('enc3', reuse=reuse_vars):
        kernel_size = [3, 3, n_filters[2]]

        enc3_conv = vgg2d(
            layers[-1],
            kernel_size=kernel_size,
            n_convolution=n_mod3,
            padding=padding,
            activation_fn=activation_fn,
            use_pooling=False,
            reuse=reuse_vars)

        layers.append(enc3_conv)
        skips.append(layers[-1])

    with tf.variable_scope('enc4', reuse=reuse_vars):
        kernel_size = [3, 3, n_filters[3]]

        enc4_conv = vgg2d(
            layers[-1],
            kernel_size=kernel_size,
            n_convolution=n_mod4,
            padding=padding,
            activation_fn=activation_fn,
            use_pooling=False,
            reuse=reuse_vars)

        layers.append(enc4_conv)
        skips.append(layers[-1])

    with tf.variable_scope('enc5', reuse=reuse_vars):
        kernel_size = [3, 3, n_filters[4]]

        enc5_conv = vgg2d(
            layers[-1],
            kernel_size=kernel_size,
            n_convolution=n_mod5,
            padding=padding,
            activation_fn=activation_fn,
            use_pooling=False,
            reuse=reuse_vars)

        layers.append(enc5_conv)

    return layers, skips


'''
Decoder architectures
'''
def fusionnet_decoder(layer,
                      skips,
                      shape,
                      activation_fn=tf.nn.relu,
                      output_fn=tf.identity,
                      reuse_vars=tf.AUTO_REUSE):
    '''
    Creates a decoder with up-convolutions to bring resolution from 1/32 to 1

    Args:
        layer : tensor (or list[tensor])
            N x H x W x D latent representation (will also handle list of layers
            for backwards compatibility)
        skips : list[tensor]
            list of skip connections
        shape : list[int]
            [H W] list of dimensions for the final output
        activation_fn : func
            activation function after convolution
        output_fn : func
            activation function to produce output predictions
        reuse_vars : bool
            if set, reuse weights if have already been defined in same variable scope
    Returns:
        list[tensor] : list containing all layers
        list[tensor] : list containing prediction and upsampled prediction at original resolution
    '''

    layers = layer if isinstance(layer, list) else [layer]
    outputs = []
    padding = 'SAME'

    with tf.variable_scope('dec4', reuse=reuse_vars):
        kernel_size = [3, 3, 256]

        # Perform up-convolution
        dec4_upconv = upconv2d(
            layers[-1],
            shape=skips[3].get_shape().as_list()[1:3],
            kernel_size=kernel_size,
            stride=1,
            activation_fn=activation_fn,
            reuse=tf.AUTO_REUSE,
            name='upconv')

        # Concatenate with skip connection
        dec4_concat = tf.concat([dec4_upconv, skips[3]], axis=-1)

        # Convolve again
        dec4_conv = slim.conv2d(
            dec4_concat,
            num_outputs=kernel_size[2],
            kernel_size=kernel_size[0:2],
            stride=1,
            activation_fn=activation_fn,
            padding=padding,
            reuse=reuse_vars,
            scope='conv1')

        layers.append(dec4_conv)

    with tf.variable_scope('dec3', reuse=reuse_vars):
        kernel_size = [3, 3, 128]

        # Perform up-convolution
        dec3_upconv = upconv2d(
            layers[-1],
            shape=skips[2].get_shape().as_list()[1:3],
            kernel_size=kernel_size,
            stride=1,
            activation_fn=activation_fn,
            reuse=tf.AUTO_REUSE,
            name='upconv')

        # Concatenate with skip connection
        dec3_concat = tf.concat([dec3_upconv, skips[2]], axis=-1)

        # Convolve again
        dec3_conv = slim.conv2d(
            dec3_concat,
            num_outputs=kernel_size[2],
            kernel_size=kernel_size[0:2],
            stride=1,
            activation_fn=activation_fn,
            padding=padding,
            reuse=reuse_vars,
            scope='conv1')

        layers.append(dec3_conv)

    with tf.variable_scope('dec2', reuse=reuse_vars):
        kernel_size = [3, 3, 64]

        # Perform up-convolution
        dec2_upconv = upconv2d(
            layers[-1],
            shape=skips[1].get_shape().as_list()[1:3],
            kernel_size=kernel_size,
            stride=1,
            activation_fn=activation_fn,
            reuse=tf.AUTO_REUSE,
            name='upconv')

        # Concatenate with skip connection
        dec2_concat = tf.concat([dec2_upconv, skips[1]], axis=-1)

        # Convolve again
        dec2_conv = slim.conv2d(
            dec2_concat,
            num_outputs=kernel_size[2],
            kernel_size=kernel_size[0:2],
            stride=1,
            activation_fn=activation_fn,
            padding=padding,
            reuse=reuse_vars,
            scope='conv1')

        layers.append(dec2_conv)

    with tf.variable_scope('dec1', reuse=reuse_vars):
        kernel_size = [3, 3, 64]

        # Perform up-convolution
        dec1_upconv = upconv2d(
            layers[-1],
            shape=skips[0].get_shape().as_list()[1:3],
            kernel_size=kernel_size,
            stride=1,
            activation_fn=activation_fn,
            reuse=tf.AUTO_REUSE,
            name='upconv')

        # Concatenate with skip connection
        layers.append(tf.concat([dec1_upconv, skips[0]], axis=-1))

        dec1_output_scale = slim.conv2d(
            layers[-1],
            num_outputs=1,
            kernel_size=kernel_size[0:2],
            stride=1,
            activation_fn=tf.nn.sigmoid,
            padding=padding,
            reuse=reuse_vars,
            scope='output_scale')

        dec1_output_residual = slim.conv2d(
            layers[-1],
            num_outputs=1,
            kernel_size=kernel_size[0:2],
            stride=1,
            activation_fn=output_fn,
            padding=padding,
            reuse=reuse_vars,
            scope='output_residual')

        dec1_output = tf.concat([
            dec1_output_scale,
            dec1_output_residual],
            axis=-1)

        outputs.append(dec1_output)

    with tf.variable_scope('dec0', reuse=reuse_vars):
        # Perform up-sampling to original resolution
        dec0_output = tf.reshape(
            tf.image.resize_nearest_neighbor(outputs[-1], shape),
            [-1, shape[0], shape[1], 2])

        layers.append(dec0_output)
        outputs.append(dec0_output)

    return layers, outputs

def scaffnet_decoder(layer,
                     skips,
                     shape,
                     n_filters=[32, 64, 96, 128, 196],
                     activation_fn=tf.nn.relu,
                     output_fn=tf.identity,
                     n_filter_output=0,
                     reuse_vars=tf.AUTO_REUSE):
    '''
    Creates a decoder with up-convolutions to bring resolution from 1/32 to 1

    Args:
        layer : tensor (or list[tensor])
            N x H x W x D latent representation (will also handle list of layers
            for backwards compatibility)
        skips : list[tensor]
            list of skip connections
        shape : list[int]
            [H W] list of dimensions for the final output
        n_filters : float
            filters for each module
        activation_fn : func
            activation function after convolution
        output_fn : func
            activation function to produce output predictions
        n_filter_output: int
            number of filters in the output layer if 0 then use upsample
        reuse_vars : bool
            if set, reuse weights if have already been defined in same variable scope
    Returns:
        list[tensor] : list containing all layers
        list[tensor] : list containing prediction and upsampled prediction at original resolution
    '''

    layers = layer if isinstance(layer, list) else [layer]

    outputs = []
    padding = 'SAME'

    n = len(skips)-1

    with tf.variable_scope('dec4', reuse=reuse_vars):
        kernel_size = [3, 3, n_filters[3]]

        # Perform up-convolution
        dec4_upconv = upconv2d(
            layers[-1],
            shape=skips[n].get_shape().as_list()[1:3],
            kernel_size=kernel_size,
            stride=1,
            activation_fn=activation_fn,
            reuse=tf.AUTO_REUSE,
            name='upconv')

        # Concatenate with skip connection
        dec4_concat = tf.concat([dec4_upconv, skips[n]], axis=-1)

        # Convolve again
        dec4_conv = slim.conv2d(
            dec4_concat,
            num_outputs=kernel_size[2],
            kernel_size=kernel_size[0:2],
            stride=1,
            activation_fn=activation_fn,
            padding=padding,
            reuse=reuse_vars,
            scope='conv1')

        layers.append(dec4_conv)

    n = n - 1

    with tf.variable_scope('dec3', reuse=reuse_vars):
        kernel_size = [3, 3, n_filters[2]]

        # Perform up-convolution
        dec3_upconv = upconv2d(
            layers[-1],
            shape=skips[n].get_shape().as_list()[1:3],
            kernel_size=kernel_size,
            stride=1,
            activation_fn=activation_fn,
            reuse=tf.AUTO_REUSE,
            name='upconv')

        # Concatenate with skip connection
        dec3_concat = tf.concat([dec3_upconv, skips[n]], axis=-1)

        # Convolve again
        dec3_conv = slim.conv2d(
            dec3_concat,
            num_outputs=kernel_size[2],
            kernel_size=kernel_size[0:2],
            stride=1,
            activation_fn=activation_fn,
            padding=padding,
            reuse=reuse_vars,
            scope='conv1')

        layers.append(dec3_conv)

    n = n - 1

    with tf.variable_scope('dec2', reuse=reuse_vars):
        kernel_size = [3, 3, n_filters[1]]

        # Perform up-convolution
        dec2_upconv = upconv2d(
            layers[-1],
            shape=skips[n].get_shape().as_list()[1:3],
            kernel_size=kernel_size,
            stride=1,
            activation_fn=activation_fn,
            reuse=tf.AUTO_REUSE,
            name='upconv')

        # Concatenate with skip connection
        dec2_concat = tf.concat([dec2_upconv, skips[n]], axis=-1)

        # Convolve again
        dec2_conv = slim.conv2d(
            dec2_concat,
            num_outputs=kernel_size[2],
            kernel_size=kernel_size[0:2],
            stride=1,
            activation_fn=activation_fn,
            padding=padding,
            reuse=reuse_vars,
            scope='conv1')

        layers.append(dec2_conv)

    n = n - 1

    with tf.variable_scope('dec1', reuse=reuse_vars):
        kernel_size = [3, 3, n_filters[1]]

        # Perform up-convolution
        dec1_upconv = upconv2d(
            layers[-1],
            shape=skips[n].get_shape().as_list()[1:3],
            kernel_size=kernel_size,
            stride=1,
            activation_fn=activation_fn,
            reuse=tf.AUTO_REUSE,
            name='upconv')

        # Concatenate with skip connection
        dec1_concat = tf.concat([dec1_upconv, skips[n]], axis=-1)

        # Convolve again
        if n_filter_output > 0:
            dec1_conv = slim.conv2d(
                dec1_concat,
                num_outputs=kernel_size[2],
                kernel_size=kernel_size[0:2],
                stride=1,
                activation_fn=activation_fn,
                padding=padding,
                reuse=reuse_vars,
                scope='conv1')

            layers.append(dec1_conv)
        else:
            dec1_output = slim.conv2d(
                dec1_concat,
                num_outputs=1,
                kernel_size=kernel_size[0:2],
                stride=1,
                activation_fn=output_fn,
                padding=padding,
                reuse=reuse_vars,
                scope='output')

            outputs.append(dec1_output)

    n = n - 1

    with tf.variable_scope('dec0', reuse=reuse_vars):
        if n_filter_output > 0:
            kernel_size = [3, 3, n_filter_output]

            # Predict at original resolution
            dec0_upconv = upconv2d(
                layers[-1],
                shape=shape,
                kernel_size=kernel_size,
                stride=1,
                activation_fn=activation_fn,
                reuse=tf.AUTO_REUSE,
                name='upconv')

            layers.append(dec0_upconv)

            if n == 0:
                # Concatenate with skip connection
                layers.append(tf.concat([layers[-1], skips[n]], axis=-1))

            # Convolve again
            dec0_conv = slim.conv2d(
                layers[-1],
                num_outputs=kernel_size[2],
                kernel_size=kernel_size[0:2],
                stride=1,
                activation_fn=activation_fn,
                padding=padding,
                reuse=reuse_vars,
                scope='conv1')

            layers.append(dec0_conv)

            # Predict at original resolution
            dec0_output = slim.conv2d(
                dec0_conv,
                num_outputs=1,
                kernel_size=kernel_size[0:2],
                stride=1,
                activation_fn=output_fn,
                padding=padding,
                reuse=reuse_vars,
                scope='output')
        else:
            # Perform up-sampling to original resolution
            dec0_output = tf.reshape(
                tf.image.resize_nearest_neighbor(outputs[-1], shape),
                [-1, shape[0], shape[1], 1])

        outputs.append(dec0_output)

    return layers, outputs
