import tensorflow as tf
import tensorflow.contrib.slim as slim


'''
Util for creating network layers and blocks
'''
def vgg2d(x,
          ksize,
          n_conv=2,
          stride=2,
          padding='SAME',
          act_fn=tf.nn.relu,
          use_pooling=False,
          reuse=tf.AUTO_REUSE,
          name=None):
    '''
    Creates a VGG block of n_conv layers

    Args:
        x : tensor
            input tensor
        ksize : list
            3 x 1 list [k, k, f] of kernel size k, number of filters f
        n_conv : int
            number of total convolutions
        stride : int
            stride size for downsampling
        padding : str
            padding on edges in case size doesn't match
        act_fn : func
            activation function after convolution
        reuse : bool
            if set, reuse weights if have already been defined in same variable scope
        name : str
            name of node in computational graph

    Returns:
        tensor : layer after VGG convolutions
    '''

    name = name + '_vgg2d_' if name is not None else 'vgg2d_'
    layers = [x]

    for n in range(n_conv - 1):
        layer_name = name + 'conv' + str(n + 1)

        conv = slim.conv2d(
            layers[-1],
            num_outputs=ksize[2],
            kernel_size=ksize[0:2],
            stride=1,
            padding=padding,
            activation_fn=act_fn,
            reuse=reuse,
            scope=layer_name)

        layers.append(conv)

    layer_name = name + 'conv' + str(n_conv)

    if use_pooling and stride > 1:
        convn = slim.conv2d(
            layers[-1],
            num_outputs=ksize[2],
            kernel_size=ksize[0:2],
            stride=1,
            padding=padding,
            activation_fn=act_fn,
            reuse=reuse,
            scope=layer_name)

        convn = slim.max_pool2d(convn, kernel_size=[2, 2], padding=padding)
    else:
        convn = slim.conv2d(
            layers[-1],
            num_outputs=ksize[2],
            kernel_size=ksize[0:2],
            stride=stride,
            padding=padding,
            activation_fn=act_fn,
            reuse=reuse,
            scope=layer_name)

    return convn

def upconv2d(x,
             shape,
             ksize,
             stride=2,
             padding='SAME',
             act_fn=tf.nn.relu,
             reuse=tf.AUTO_REUSE,
             name=None):
    '''
    Creates a 2D up-convolution layer upsample and convolution

    Args:
        x : tensor
            input tensor
        shape : list
            2 element list of tensor y-x shape
        ksize : list
            3 x 1 list [k, k, f] of kernel size k, number of filters f
        stride : int
            stride size of convolution
        padding : str
            padding on edges in case size doesn't match
        act_fn : func
            activation function after convolution
        reuse : bool
            if set, reuse weights if have already been defined in same variable scope
        name : str
            name of node in computational graph

    Returns:
        tensor : layer after performing up-convolution
    '''

    name = name if name is not None else ''
    layer_name = name + 'upconv'

    x_up = tf.image.resize_nearest_neighbor(x, shape)

    conv = slim.conv2d(
        x_up,
        num_outputs=ksize[2],
        kernel_size=ksize[0:2],
        stride=stride,
        padding=padding,
        activation_fn=act_fn,
        reuse=reuse,
        scope=layer_name)

    return conv

def aspp2d(x,
           n_output,
           n_conv=1,
           rates=[1, 2, 4, 8],
           filters=[8, 8, 8, 8],
           padding='SAME',
           act_fn=tf.nn.relu,
           reuse=tf.AUTO_REUSE,
           name=None):
    '''
    Creates atrous pyramid with rates for dilated convolutions

    Args:
        x : tensor
            input tensor
        n_output : int
            number of output filters
        n_conv : int
            number of 1 x 1 convolutions
        rates : list of int
            dilation rates for atrous convolution following first convolution
        filters : list of int
            number of filters for atrous convolutions
        padding : str
            padding on edges in case size doesn't match
        act_fn : func
            activation function after convolution
        reuse : bool
            if set, reuse weights if have already been defined in same variable scope
        name : str
            name of node in computational graph

    Returns:
        tensor : layer after atrous pyramid
    '''

    name = name + '_aspp2d_' if name is not None else 'aspp2d_'
    layers = [x]

    for n in range(len(rates)):
        layer_name = name + 'conv' + str(n + 1)

        conv = slim.conv2d(
            x,
            num_outputs=filters[n],
            kernel_size=[3, 3],
            stride=1,
            rate=rates[n],
            padding=padding,
            activation_fn=act_fn,
            reuse=reuse,
            scope=layer_name)

        layers.append(conv)

    layers.append(tf.concat(layers, axis=-1))

    for m in range(n_conv):
        layer_name = name + 'conv' + str(n + m + 2)

        conv = slim.conv2d(
            layers[-1],
            num_outputs=n_output,
            kernel_size=[1, 1],
            stride=1,
            padding=padding,
            activation_fn=act_fn,
            reuse=reuse,
            scope=layer_name)

        layers.append(conv)

    return layers[-1]

def spp2d(x,
          n_output,
          n_conv=1,
          rates=[2, 5],
          keep_input=False,
          padding='SAME',
          act_fn=tf.nn.relu,
          reuse=tf.AUTO_REUSE,
          name=None):
    '''
    Creates spatial pyramid pooling with pooling rates and convolves the pyramid

    Args:
        x : tensor
            input tensor
        n_output : int
            number of output filters
        n_conv : int
            number of 1 x 1 convolutions
        rates : list of int
            pooling rates for max pooling
        keep_input : bool
            preserves the input values in the feature maps
        padding : str
            padding on edges in case size doesn't match
        act_fn : func
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

    if keep_input:
        valid = tf.expand_dims(x[..., 1], axis=-1)

    for r in rates:
        layer = slim.max_pool2d(x, kernel_size=[r, r], stride=1, padding=padding)

        if keep_input:
            layer = layer * (-1.0 * valid + 1.0) + x

        layers.append(layer)

    layers.append(tf.concat(layers, axis=-1))

    for n in range(n_conv):
        layer_name = name + 'conv' + str(n + 1)

        conv = slim.conv2d(
            layers[-1],
            num_outputs=n_output,
            kernel_size=[1, 1],
            stride=1,
            padding=padding,
            activation_fn=act_fn,
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

def connet16(depth,
             n_output=1,
             act_fn=tf.nn.relu,
             out_fn=tf.identity,
             pool_rates_spp=[5, 7, 9, 11],
             n_conv_spp=1,
             keep_input_spp=False,
             n_filter_output=0):
    '''
    Creates connectivity network with 16 initial filters

    Args:
        depth : tensor
            input sparse depth with validity map (N x H x W x 2)
        n_output : int
            number of output channels
        act_fn : func
            activation function after convolution
        out_fn : func
            activation function to produce output predictions
        pool_rates_spp : list of int
            sets the max pooling rates for spatial pyramid pooling
        n_conv_spp : int
            number of 1 x 1 convolutions in SPP
        keep_input_spp : bool
            preserves the input values in the feature maps
        n_filter_output: int
            number of filters in the output layer if 0 then use upsample

    Returns:
        list : list containing prediction and upsampled prediction at original resolution
    '''

    with tf.variable_scope('scaffnet16'):
        n_filters = [16, 32, 64, 96, 128]
        shape = depth.get_shape().as_list()[1:3]

        layers, skips = connectivity_encoder(
            depth,
            n_mod1=1, n_mod2=1, n_mod3=1, n_mod4=1, n_mod5=1,
            n_filters=n_filters,
            act_fn=act_fn,
            pool_rates_spp=pool_rates_spp,
            keep_input_spp=keep_input_spp,
            n_conv_spp=n_conv_spp)

        layers, outputs = connectivity_decoder(
            layers,
            skips,
            shape,
            n_output=n_output,
            n_filters=n_filters,
            act_fn=act_fn,
            out_fn=out_fn,
            n_filter_output=n_filter_output)

        return outputs

def connet32(depth,
             n_output=1,
             act_fn=tf.nn.relu,
             out_fn=tf.identity,
             pool_rates_spp=[5, 7, 9, 11],
             n_conv_spp=1,
             keep_input_spp=False,
             n_filter_output=0):
    '''
    Creates connectivity network with 32 initial filters

    Args:
        depth : tensor
            input sparse depth with validity map (N x H x W x 2)
        n_output : int
            number of output channels
        act_fn : func
            activation function after convolution
        out_fn : func
            activation function to produce output predictions
        pool_rates_spp : list of int
            sets the max pooling rates for spatial pyramid pooling
        n_conv_spp : int
            number of 1 x 1 convolutions in SPP
        keep_input_spp : bool
            preserves the input values in the feature maps
        n_filter_output: int
            number of filters in the output layer if 0 then use upsample

    Returns:
        list : list containing prediction and upsampled prediction at original resolution
    '''

    with tf.variable_scope('scaffnet32'):
        n_filters = [32, 64, 96, 128, 196]
        shape = depth.get_shape().as_list()[1:3]

        layers, skips = connectivity_encoder(
            depth,
            n_mod1=1,
            n_mod2=1,
            n_mod3=1,
            n_mod4=1,
            n_mod5=1,
            n_filters=n_filters,
            act_fn=act_fn,
            pool_rates_spp=pool_rates_spp,
            n_conv_spp=n_conv_spp,
            keep_input_spp=keep_input_spp)

        layers, outputs = connectivity_decoder(
            layers,
            skips,
            shape,
            n_output=n_output,
            n_filters=n_filters,
            act_fn=act_fn,
            out_fn=out_fn,
            n_filter_output=n_filter_output)

        return outputs

def vggnet08(image,
             depth,
             n_output=1,
             act_fn=tf.nn.relu,
             out_fn=tf.identity,
             image_filter_pct=0.75,
             depth_filter_pct=0.25,
             n_filter_dec0=0):
    '''
    Creates a VGG08 late fusion network

    Args:
        image : tensor
            input image (N x H x W x D)
        depth : tensor
            input sdepth with validity map (N x H x W x 2)
        n_output : int
            number of output channels
        act_fn : func
            activation function after convolution
        out_fn : func
            activation function to produce output predictions
        im_filter_pct : float
            percent of parameters to allocate to the image branch
        zv_filter_pct : float
            percent of parameters to allocate to the depth branch
        n_filter_dec0 : int
            number of filters in decoder layer 0
        reuse_vars : bool
            if set, reuse weights if have already been defined in same variable scope

    Returns:
        list : list containing prediction and upsampled prediction at original resolution
    '''

    with tf.variable_scope('vggnet08'):
        shape = image.get_shape().as_list()[1:3]

        layers, skips = vggnet_encoder(
            image,
            depth,
            n_mod1=1,
            n_mod2=1,
            n_mod3=1,
            n_mod4=1,
            n_mod5=1,
            act_fn=act_fn,
            image_filter_pct=image_filter_pct,
            depth_filter_pct=depth_filter_pct)

        layers, outputs = decoder(
            layers,
            skips,
            shape,
            n_output=n_output,
            n_filter_dec0=n_filter_dec0,
            act_fn=act_fn,
            out_fn=out_fn)

        return outputs

def vggnet11(image,
             depth,
             n_output=1,
             act_fn=tf.nn.relu,
             out_fn=tf.identity,
             image_filter_pct=0.75,
             depth_filter_pct=0.25,
             n_filter_dec0=0):
    '''
    Creates a VGG11 late fusion network

    Args:
        image : tensor
            input image (N x H x W x D)
        depth : tensor
            input sparse depth with validity map (N x H x W x 2)
        n_output : int
            number of output channels
        act_fn : func
            activation function after convolution
        out_fn : func
            activation function to produce output predictions
        im_filter_pct : float
            percent of parameters to allocate to the image branch
        zv_filter_pct : float
            percent of parameters to allocate to the depth branch
        n_filter_dec0 : int
            number of filters in decoder layer 0
        reuse_vars : bool
            if set, reuse weights if have already been defined in same variable scope

    Returns:
        list : list containing prediction and upsampled prediction at original resolution
    '''
    with tf.variable_scope('vggnet11'):
        shape = image.get_shape().as_list()[1:3]

        layers, skips = vggnet_encoder(
            image,
            depth,
            n_mod1=1, n_mod2=1, n_mod3=2, n_mod4=2, n_mod5=2,
            act_fn=act_fn,
            image_filter_pct=image_filter_pct,
            depth_filter_pct=depth_filter_pct)

        layers, outputs = decoder(
            layers,
            skips,
            shape,
            n_output=n_output,
            n_filter_dec0=n_filter_dec0,
            act_fn=act_fn,
            out_fn=out_fn)

        return outputs


'''
Encoder architectures
'''
def vggnet_encoder(image,
                   depth,
                   n_mod1=1,
                   n_mod2=2,
                   n_mod3=2,
                   n_mod4=2,
                   n_mod5=2,
                   image_filter_pct=0.75,
                   depth_filter_pct=0.25,
                   act_fn=tf.nn.relu,
                   reuse_vars=tf.AUTO_REUSE):
    '''
    Creates an early or late fusion (two branches, one for processing image and the other depth)
    VGGnet encoder with 5 VGG blocks each with resolution 1 -> 1/32

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
        act_fn : func
            activation function after convolution
        reuse_vars : bool
            if set, reuse weights if have already been defined in same variable scope

    Returns:
        list : list containing all the layers last element is the
            latent representation (1/32 resolution)
        list : list containing all the skip connections
    '''

    layers = []
    skips  = []
    padding = 'SAME'

    with tf.variable_scope('enc1', reuse=reuse_vars):
        ksize = [5, 5, 64]

        enc1_conv_image = vgg2d(
            image,
            ksize=ksize[0:2] + [int(image_filter_pct * ksize[2])],
            n_conv=n_mod1,
            padding=padding,
            act_fn=act_fn,
            reuse=reuse_vars,
            name='im')

        enc1_conv_depth = vgg2d(
            depth,
            ksize=ksize[0:2] + [int(depth_filter_pct * ksize[2])],
            n_conv=n_mod1,
            padding=padding,
            act_fn=act_fn,
            reuse=reuse_vars,
            name='zv')

        layers.append(enc1_conv_depth)
        layers.append(enc1_conv_image)
        skips.append(tf.concat([enc1_conv_depth, enc1_conv_image], axis=-1))

    with tf.variable_scope('enc2', reuse=reuse_vars):
        ksize = [3, 3, 128]

        enc2_conv_image = vgg2d(
            layers[-1],
            ksize=ksize[0:2] + [int(image_filter_pct * ksize[2])],
            n_conv=n_mod2,
            padding=padding,
            act_fn=act_fn,
            reuse=reuse_vars,
            name='im')

        enc2_conv_depth = vgg2d(
            layers[-2],
            ksize=ksize[0:2] + [int(depth_filter_pct * ksize[2])],
            n_conv=n_mod2,
            padding=padding,
            act_fn=act_fn,
            reuse=reuse_vars,
            name='zv')

        layers.append(enc2_conv_depth)
        layers.append(enc2_conv_image)
        skips.append(tf.concat([enc2_conv_depth, enc2_conv_image], axis=-1))

    with tf.variable_scope('enc3', reuse=reuse_vars):
        ksize = [3, 3, 256]

        enc3_conv_image = vgg2d(
            layers[-1],
            ksize=ksize[0:2]+[int(image_filter_pct * ksize[2])],
            n_conv=n_mod3,
            padding=padding,
            act_fn=act_fn,
            reuse=reuse_vars,
            name='im')

        enc3_conv_depth = vgg2d(
            layers[-2],
            ksize=ksize[0:2] + [int(depth_filter_pct * ksize[2])],
            n_conv=n_mod3,
            padding=padding,
            act_fn=act_fn,
            reuse=reuse_vars,
            name='zv')

        layers.append(enc3_conv_depth)
        layers.append(enc3_conv_image)
        skips.append(tf.concat([enc3_conv_depth, enc3_conv_image], axis=-1))

    with tf.variable_scope('enc4', reuse=reuse_vars):
        ksize = [3, 3, 512]

        enc4_conv_image = vgg2d(
            layers[-1],
            ksize=ksize[0:2] + [int(image_filter_pct * ksize[2])],
            n_conv=n_mod4,
            padding=padding,
            act_fn=act_fn,
            reuse=reuse_vars,
            name='im')

        enc4_conv_depth = vgg2d(
            layers[-2],
            ksize=ksize[0:2] + [int(depth_filter_pct * ksize[2])],
            n_conv=n_mod4,
            padding=padding,
            act_fn=act_fn,
            reuse=reuse_vars,
            name='zv')

        layers.append(enc4_conv_depth)
        layers.append(enc4_conv_image)
        skips.append(tf.concat([enc4_conv_depth, enc4_conv_image], axis=-1))

    with tf.variable_scope('enc5', reuse=reuse_vars):
        ksize = [3, 3, 512]

        enc5_conv_image = vgg2d(
            layers[-1],
            ksize=ksize[0:2] + [int(image_filter_pct * ksize[2])],
            n_conv=n_mod5,
            padding=padding,
            act_fn=act_fn,
            reuse=reuse_vars,
            name='im')

        enc5_conv_depth = vgg2d(
            layers[-2],
            ksize=ksize[0:2] + [int(depth_filter_pct * ksize[2])],
            n_conv=n_mod5,
            padding=padding,
            act_fn=act_fn,
            reuse=reuse_vars,
            name='zv')

        layers.append(tf.concat([enc5_conv_depth, enc5_conv_image], axis=-1))

    return layers, skips

def connectivity_encoder(depth,
                         n_mod1=1,
                         n_mod2=1,
                         n_mod3=1,
                         n_mod4=1,
                         n_mod5=1,
                         n_filters=[32, 64, 96, 128, 196],
                         act_fn=tf.nn.relu,
                         pool_rates_spp=[5, 7, 9, 11],
                         n_conv_spp=3,
                         keep_input_spp=False,
                         reuse_vars=tf.AUTO_REUSE):
    '''
    Creates a shallow connectivity network 1 -> 1/32

    Args:
        depth : tensor
            input sparse depth with validity map (N x H x W x 2)
        n_mod<n> : int
            number of convolutional layers to perform in nth VGG block
        n_filter : float
            filters for each module
        act_fn : func
            activation function after convolution
        pool_rates_spp : list of int
            sets the max pooling rates for spatial pyramid pooling
        n_conv_spp : int
            number of 1 x 1 convolutions in SPP
        keep_input_spp : bool
            if set then preserves the input values in feature maps when performing SPP
        reuse_vars : bool
            if set, reuse weights if have already been defined in same variable scope

    Returns:
        list : list containing all the layers last element is the
            latent representation (1/32 resolution)
        list : list containing all the skip connections
    '''
    layers = []
    skips  = []
    padding = 'SAME'
    with tf.variable_scope('enc1', reuse=reuse_vars):
        ksize = [5, 5, n_filters[0]]

        if len(pool_rates_spp) > 0:
            spatial_pyramid_pooling = spp2d(
                depth,
                n_output=ksize[2],
                n_conv=n_conv_spp,
                rates=pool_rates_spp,
                keep_input=keep_input_spp,
                padding=padding,
                act_fn=act_fn,
                reuse=reuse_vars)

            layers.append(spatial_pyramid_pooling)

            enc1_conv = vgg2d(
                layers[-1],
                ksize=ksize,
                n_conv=1,
                stride=2,
                padding=padding,
                act_fn=act_fn,
                use_pooling=False,
                reuse=reuse_vars)

            layers.append(enc1_conv)
        else:
            enc1_conv = vgg2d(
                depth,
                ksize=ksize,
                n_conv=n_mod1,
                padding=padding,
                act_fn=act_fn,
                use_pooling=False,
                reuse=reuse_vars)

            layers.append(enc1_conv)

        skips.append(layers[-1])

    with tf.variable_scope('enc2', reuse=reuse_vars):
        ksize = [3, 3, n_filters[1]]

        enc2_conv = vgg2d(
            layers[-1],
            ksize=ksize,
            n_conv=n_mod2,
            padding=padding,
            act_fn=act_fn,
            use_pooling=False,
            reuse=reuse_vars)

        layers.append(enc2_conv)
        skips.append(layers[-1])

    with tf.variable_scope('enc3', reuse=reuse_vars):
        ksize = [3, 3, n_filters[2]]

        enc3_conv = vgg2d(
            layers[-1],
            ksize=ksize,
            n_conv=n_mod3,
            padding=padding,
            act_fn=act_fn,
            use_pooling=False,
            reuse=reuse_vars)

        layers.append(enc3_conv)
        skips.append(layers[-1])

    with tf.variable_scope('enc4', reuse=reuse_vars):
        ksize = [3, 3, n_filters[3]]

        enc4_conv = vgg2d(
            layers[-1],
            ksize=ksize,
            n_conv=n_mod4,
            padding=padding,
            act_fn=act_fn,
            use_pooling=False,
            reuse=reuse_vars)

        layers.append(enc4_conv)
        skips.append(layers[-1])

    with tf.variable_scope('enc5', reuse=reuse_vars):
        ksize = [3, 3, n_filters[4]]

        enc5_conv = vgg2d(
            layers[-1],
            ksize=ksize,
            n_conv=n_mod5,
            padding=padding,
            act_fn=act_fn,
            use_pooling=False,
            reuse=reuse_vars)

        layers.append(enc5_conv)

    return layers, skips


'''
Decoder architectures
'''
def decoder(layer,
            skips,
            shape,
            n_output=1,
            n_filter_dec0=0,
            act_fn=tf.nn.relu,
            out_fn=tf.identity,
            reuse_vars=tf.AUTO_REUSE):
    '''
    Creates a decoder with up-convolves the latent representation to 5 times the
    resolution from 1/32 -> 1

    Args:
        layer : tensor (or list)
            N x H x W x D latent representation (will also handle list of layers
            for backwards compatibility)
        skips : list
            list of skip connections
        shape : list
            [H W] list of dimensions for the final output
        n_output : int
            number of channels in output
        n_filter_dec0 : int
            number of filters in decoder layer 0
        act_fn : func
            activation function after convolution
        out_fn : func
            activation function to produce output predictions
        reuse_vars : bool
            if set, reuse weights if have already been defined in same variable scope

    Returns:
        list : list containing all layers
        list : list containing prediction and upsampled prediction at original resolution
    '''

    layers = layer if isinstance(layer, list) else [layer]
    outputs = []
    padding = 'SAME'

    with tf.variable_scope('dec4', reuse=reuse_vars):
        ksize = [3, 3, 256]

        # Perform up-convolution
        dec4_upconv = upconv2d(
            layers[-1],
            shape=skips[3].get_shape().as_list()[1:3],
            ksize=ksize,
            stride=1,
            act_fn=act_fn,
            reuse=tf.AUTO_REUSE)

        # Concatenate with skip connection
        dec4_concat = tf.concat([dec4_upconv, skips[3]], axis=-1)

        # Convolve again
        dec4_conv = slim.conv2d(
            dec4_concat,
            num_outputs=ksize[2],
            kernel_size=ksize[0:2],
            stride=1,
            activation_fn=act_fn,
            padding=padding,
            reuse=reuse_vars,
            scope='conv1')

        layers.append(dec4_conv)

    with tf.variable_scope('dec3', reuse=reuse_vars):
        ksize = [3, 3, 128]

        # Perform up-convolution
        dec3_upconv = upconv2d(
            layers[-1],
            shape=skips[2].get_shape().as_list()[1:3],
            ksize=ksize,
            stride=1,
            act_fn=act_fn,
            reuse=tf.AUTO_REUSE,
            name='upconv')

        # Concatenate with skip connection
        dec3_concat = tf.concat([dec3_upconv, skips[2]], axis=-1)

        # Convolve again
        dec3_conv = slim.conv2d(
            dec3_concat,
            num_outputs=ksize[2],
            kernel_size=ksize[0:2],
            stride=1,
            activation_fn=act_fn,
            padding=padding,
            reuse=reuse_vars,
            scope='conv1')

        layers.append(dec3_conv)

    with tf.variable_scope('dec2', reuse=reuse_vars):
        ksize = [3, 3, 64]

        # Perform up-convolution
        dec2_upconv = upconv2d(
            layers[-1],
            shape=skips[1].get_shape().as_list()[1:3],
            ksize=ksize,
            stride=1,
            act_fn=act_fn,
            reuse=tf.AUTO_REUSE,
            name='upconv')

        # Concatenate with skip connection
        dec2_concat = tf.concat([dec2_upconv, skips[1]], axis=-1)

        # Convolve again
        dec2_conv = slim.conv2d(
            dec2_concat,
            num_outputs=ksize[2],
            kernel_size=ksize[0:2],
            stride=1,
            activation_fn=act_fn,
            padding=padding,
            reuse=reuse_vars,
            scope='conv1')

        layers.append(dec2_conv)

    with tf.variable_scope('dec1', reuse=reuse_vars):
        ksize = [3, 3, 64]

        # Perform up-convolution
        dec1_upconv = upconv2d(
            layers[-1],
            shape=skips[0].get_shape().as_list()[1:3],
            ksize=ksize,
            stride=1,
            act_fn=act_fn,
            reuse=tf.AUTO_REUSE,
            name='upconv')

        # Concatenate with skip connection
        layers.append(tf.concat([dec1_upconv, skips[0]], axis=-1))

        if n_filter_dec0 > 0:
            # Convolve again
            dec1_conv = slim.conv2d(
                layers[-1],
                num_outputs=ksize[2],
                kernel_size=ksize[0:2],
                stride=1,
                activation_fn=act_fn,
                padding=padding,
                reuse=reuse_vars,
                scope='conv1')

            layers.append(dec1_conv)
        else:
            # Convolve again
            if n_output == 1:
                dec1_output = slim.conv2d(
                    layers[-1],
                    num_outputs=n_output,
                    kernel_size=ksize[0:2],
                    stride=1,
                    activation_fn=out_fn,
                    padding=padding,
                    reuse=reuse_vars,
                    scope='output')

                outputs.append(dec1_output)
            elif n_output == 2:
                dec1_output_scale = slim.conv2d(
                    layers[-1],
                    num_outputs=1,
                    kernel_size=ksize[0:2],
                    stride=1,
                    activation_fn=tf.nn.sigmoid,
                    padding=padding,
                    reuse=reuse_vars,
                    scope='scale_output')

                dec1_output_residual = slim.conv2d(
                    layers[-1],
                    num_outputs=1,
                    kernel_size=ksize[0:2],
                    stride=1,
                    activation_fn=out_fn,
                    padding=padding,
                    reuse=reuse_vars,
                    scope='residual_output')

                dec1_output = tf.concat([
                    dec1_output_scale,
                    dec1_output_residual],
                    axis=-1)

                outputs.append(dec1_output)

    with tf.variable_scope('dec0', reuse=reuse_vars):
        if n_filter_dec0 > 0:
            ksize = [3, 3, n_filter_dec0]

            # Predict at original resolution
            dec0_upconv = upconv2d(
                layers[-1],
                shape=shape,
                ksize=ksize,
                stride=1,
                act_fn=act_fn,
                reuse=tf.AUTO_REUSE,
                name='upconv')

            layers.append(dec0_upconv)

            dec0_conv = slim.conv2d(
                layers[-1],
                num_outputs=ksize[2],
                kernel_size=ksize[0:2],
                stride=1,
                activation_fn=act_fn,
                padding=padding,
                reuse=reuse_vars,
                scope='conv1')

            layers.append(dec0_conv)

            if n_output == 1:
                dec0_output = slim.conv2d(
                    layers[-1],
                    num_outputs=n_output,
                    kernel_size=ksize[0:2],
                    stride=1,
                    activation_fn=out_fn,
                    padding=padding,
                    reuse=reuse_vars,
                    scope='output')

                outputs.append(dec0_output)

            elif n_output == 2:
                dec0_output_scale = slim.conv2d(
                    layers[-1],
                    num_outputs=1,
                    kernel_size=ksize[0:2],
                    stride=1,
                    activation_fn=tf.nn.sigmoid,
                    padding=padding,
                    reuse=reuse_vars,
                    scope='scale_output')

                dec0_output_residual = slim.conv2d(
                    layers[-1],
                    num_outputs=1,
                    kernel_size=ksize[0:2],
                    stride=1,
                    activation_fn=out_fn,
                    padding=padding,
                    reuse=reuse_vars,
                    scope='residual_output')

                dec0_output = tf.concat([
                    dec0_output_scale,
                    dec0_output_residual],
                    axis=-1)

                outputs.append(dec0_output)

        else:
            # Perform up-sampling to original resolution
            dec0_output = tf.reshape(
                tf.image.resize_nearest_neighbor(outputs[-1], shape),
                [-1, shape[0], shape[1], n_output])

            layers.append(dec0_output)
            outputs.append(dec0_output)

    return layers, outputs

def connectivity_decoder(layer,
                         skips,
                         shape,
                         n_output=1,
                         n_filters=[32, 64, 96, 128, 196],
                         act_fn=tf.nn.relu,
                         out_fn=tf.identity,
                         n_filter_output=0,
                         reuse_vars=tf.AUTO_REUSE):
    '''
    Creates a decoder with up-convolves the latent representation to 5 times the
    resolution from 1/32 -> 1

    Args:
        layer : tensor (or list)
            N x H x W x D latent representation (will also handle list of layers
            for backwards compatibility)
        skips : list
            list of skip connections
        shape : list
            [H W] list of dimensions for the final output
        n_output : int
            number of channels in output
        n_filters : float
            filters for each module
        act_fn : func
            activation function after convolution
        out_fn : func
            activation function to produce output predictions
        n_filter_output: int
            number of filters in the output layer if 0 then use upsample
        reuse_vars : bool
            if set, reuse weights if have already been defined in same variable scope
    Returns:
        list : list containing all layers
        list : list containing prediction and upsampled prediction at original resolution
    '''

    layers = layer if isinstance(layer, list) else [layer]

    outputs = []
    padding = 'SAME'
    n = len(skips)-1

    with tf.variable_scope('dec4', reuse=reuse_vars):
        ksize = [3, 3, n_filters[3]]

        # Perform up-convolution
        dec4_upconv = upconv2d(
            layers[-1],
            shape=skips[n].get_shape().as_list()[1:3],
            ksize=ksize,
            stride=1,
            act_fn=act_fn,
            reuse=tf.AUTO_REUSE)

        # Concatenate with skip connection
        dec4_concat = tf.concat([dec4_upconv, skips[n]], axis=-1)

        # Convolve again
        dec4_conv = slim.conv2d(
            dec4_concat,
            num_outputs=ksize[2],
            kernel_size=ksize[0:2],
            stride=1,
            activation_fn=act_fn,
            padding=padding,
            reuse=reuse_vars,
            scope='conv1')

        layers.append(dec4_conv)

    n = n - 1
    with tf.variable_scope('dec3', reuse=reuse_vars):
        ksize = [3, 3, n_filters[2]]

        # Perform up-convolution
        dec3_upconv = upconv2d(
            layers[-1],
            shape=skips[n].get_shape().as_list()[1:3],
            ksize=ksize,
            stride=1,
            act_fn=act_fn,
            reuse=tf.AUTO_REUSE,
            name='upconv')

        # Concatenate with skip connection
        dec3_concat = tf.concat([dec3_upconv, skips[n]], axis=-1)

        # Convolve again
        dec3_conv = slim.conv2d(
            dec3_concat,
            num_outputs=ksize[2],
            kernel_size=ksize[0:2],
            stride=1,
            activation_fn=act_fn,
            padding=padding,
            reuse=reuse_vars,
            scope='conv1')

        layers.append(dec3_conv)

    n = n - 1
    with tf.variable_scope('dec2', reuse=reuse_vars):
        ksize = [3, 3, n_filters[1]]

        # Perform up-convolution
        dec2_upconv = upconv2d(
            layers[-1],
            shape=skips[n].get_shape().as_list()[1:3],
            ksize=ksize,
            stride=1,
            act_fn=act_fn,
            reuse=tf.AUTO_REUSE,
            name='upconv')

        # Concatenate with skip connection
        dec2_concat = tf.concat([dec2_upconv, skips[n]], axis=-1)

        # Convolve again
        dec2_conv = slim.conv2d(
            dec2_concat,
            num_outputs=ksize[2],
            kernel_size=ksize[0:2],
            stride=1,
            activation_fn=act_fn,
            padding=padding,
            reuse=reuse_vars,
            scope='conv1')

        layers.append(dec2_conv)

    n = n-1
    with tf.variable_scope('dec1', reuse=reuse_vars):
        ksize = [3, 3, n_filters[1]]

        # Perform up-convolution
        dec1_upconv = upconv2d(
            layers[-1],
            shape=skips[n].get_shape().as_list()[1:3],
            ksize=ksize,
            stride=1,
            act_fn=act_fn,
            reuse=tf.AUTO_REUSE,
            name='upconv')

        # Concatenate with skip connection
        dec1_concat = tf.concat([dec1_upconv, skips[n]], axis=-1)

        # Convolve again
        if n_filter_output > 0:
            dec1_conv = slim.conv2d(
                dec1_concat,
                num_outputs=ksize[2],
                kernel_size=ksize[0:2],
                stride=1,
                activation_fn=act_fn,
                padding=padding,
                reuse=reuse_vars,
                scope='conv1')

            layers.append(dec1_conv)
        else:
            dec1_output = slim.conv2d(
                dec1_concat,
                num_outputs=n_output,
                kernel_size=ksize[0:2],
                stride=1,
                activation_fn=out_fn,
                padding=padding,
                reuse=reuse_vars,
                scope='output')

            outputs.append(dec1_output)

    n = n - 1
    with tf.variable_scope('dec0', reuse=reuse_vars):
        if n_filter_output > 0:
            ksize = [3, 3, n_filter_output]

            # Predict at original resolution
            dec0_upconv = upconv2d(
                layers[-1],
                shape=shape,
                ksize=ksize,
                stride=1,
                act_fn=act_fn,
                reuse=tf.AUTO_REUSE,
                name='upconv')

            layers.append(dec0_upconv)

            if n == 0:
                # Concatenate with skip connection
                layers.append(tf.concat([layers[-1], skips[n]], axis=-1))

            # Convolve again
            dec0_conv = slim.conv2d(
                layers[-1],
                num_outputs=ksize[2],
                kernel_size=ksize[0:2],
                stride=1,
                activation_fn=act_fn,
                padding=padding,
                reuse=reuse_vars,
                scope='conv1')

            layers.append(dec0_conv)

            # Predict at original resolution
            dec0_output = slim.conv2d(
                dec0_conv,
                num_outputs=n_output,
                kernel_size=ksize[0:2],
                stride=1,
                activation_fn=out_fn,
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
