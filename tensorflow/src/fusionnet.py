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
import os, time
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

import global_constants as settings
import data_utils, eval_utils
from fusionnet_dataloader import FusionNetDataloader
from fusionnet_model import FusionNetModel
from log_utils import log


def train(train_image_path,
          train_input_depth_path,
          train_sparse_depth_path,
          train_intrinsics_path,
          # Validation data filepaths
          val_image_path=None,
          val_input_depth_path=None,
          val_sparse_depth_path=None,
          val_ground_truth_path=None,
          # Dataloader settings
          n_batch=settings.N_BATCH,
          n_height=settings.N_HEIGHT,
          n_width=settings.N_WIDTH,
          crop_type=settings.CROP_TYPE,
          augmentation_random_horizontal_crop=False,
          augmentation_random_vertical_crop=False,
          # Network settings
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
          # Training settings
          n_epoch=settings.N_EPOCH,
          learning_rates=settings.LEARNING_RATES,
          learning_schedule=settings.LEARNING_SCHEDULE,
          # Loss function settings
          validity_map_color=settings.VALIDITY_MAP_COLOR,
          w_color=settings.W_COLOR,
          w_structure=settings.W_STRUCTURE,
          w_sparse_depth=settings.W_SPARSE_DEPTH,
          w_smoothness=settings.W_SMOOTHNESS,
          w_prior_depth=settings.W_PRIOR_DEPTH,
          residual_threshold_prior_depth=settings.RESIDUAL_THRESHOLD_PRIOR_DEPTH,
          rotation_param=settings.ROTATION_PARAM,
          # Depth evaluation settings
          min_evaluate_depth=settings.MIN_EVALUATE_DEPTH,
          max_evaluate_depth=settings.MAX_EVALUATE_DEPTH,
          # Checkpoint settings
          n_checkpoint=settings.N_CHECKPOINT,
          n_summary=settings.N_SUMMARY,
          checkpoint_path=settings.CHECKPOINT_PATH,
          restore_path=settings.RESTORE_PATH,
          # Hardware settings
          n_thread=settings.N_THREAD):

    model_path = os.path.join(checkpoint_path, 'model.ckpt')
    event_path = os.path.join(checkpoint_path, 'events')
    log_path = os.path.join(checkpoint_path, 'results.txt')

    # Load image, input depth, sparse depth, instrinsics paths from file
    train_image_paths = data_utils.read_paths(train_image_path)
    train_input_depth_paths = data_utils.read_paths(train_input_depth_path)
    train_sparse_depth_paths = data_utils.read_paths(train_sparse_depth_path)
    train_intrinsics_paths = data_utils.read_paths(train_intrinsics_path)

    n_train_sample = len(train_image_paths)

    assert n_train_sample == len(train_input_depth_paths)
    assert n_train_sample == len(train_sparse_depth_paths)
    assert n_train_sample == len(train_intrinsics_paths)

    n_train_step = n_epoch * np.ceil(n_train_sample / n_batch).astype(np.int32)

    # Load image, input depth, and sparse depth paths from file for validation
    val_image_paths = data_utils.read_paths(val_image_path)
    val_input_depth_paths = data_utils.read_paths(val_input_depth_path)
    val_sparse_depth_paths = data_utils.read_paths(val_sparse_depth_path)
    val_ground_truth_paths = data_utils.read_paths(val_ground_truth_path)

    n_val_sample = len(val_image_paths)

    assert n_val_sample == len(val_input_depth_paths)
    assert n_val_sample == len(val_sparse_depth_paths)
    assert n_val_sample == len(val_ground_truth_paths)

    val_image_paths = data_utils.pad_batch(val_image_paths, n_batch)
    val_input_depth_paths = data_utils.pad_batch(val_input_depth_paths, n_batch)
    val_sparse_depth_paths = data_utils.pad_batch(val_sparse_depth_paths, n_batch)

    # Load validation ground truth and do center crop
    val_ground_truths = []
    for idx in range(len(val_ground_truth_paths)):
        ground_truth, validity_map_ground_truth = \
            data_utils.load_depth_with_validity_map(val_ground_truth_paths[idx])

        ground_truth = np.concatenate([
            np.expand_dims(ground_truth, axis=-1),
            np.expand_dims(validity_map_ground_truth, axis=-1)],
            axis=-1)

        # Get start and end of crop
        if crop_type == 'center':
            start_height = int(float(ground_truth.shape[0] - n_height))
        elif crop_type == 'bottom':
            start_height = ground_truth.shape[0] - n_height
        else:
            start_height = 0

        end_height = n_height + start_height

        start_width = int(float(ground_truth.shape[1] - n_width) / 2.0)
        end_width = n_width + start_width

        ground_truth = \
            ground_truth[start_height:end_height, start_width:end_width, :]

        val_ground_truths.append(ground_truth)

    val_ground_truth_paths = data_utils.pad_batch(val_ground_truth_paths, n_batch)

    with tf.Graph().as_default():
        # Set up current training step
        global_step = tf.Variable(0, trainable=False)

        # Initialize optimizer with learning schedule
        learning_schedule_steps = [
            np.int32((float(v) / n_epoch) * n_train_step) for v in learning_schedule
        ]
        learning_rate_schedule = tf.train.piecewise_constant(
            global_step,
            learning_schedule_steps,
            learning_rates)

        optimizer = tf.train.AdamOptimizer(learning_rate_schedule)

        # Initialize dataloader
        dataloader = FusionNetDataloader(
            shape=[n_batch, n_height, n_width, 3],
            name='fusionnet_dataloader',
            is_training=True,
            n_thread=n_thread,
            prefetch_size=2 * n_thread)

        # Fetch the input from dataloader
        image0 = dataloader.next_element[0]
        image1 = dataloader.next_element[1]
        image2 = dataloader.next_element[2]
        input_depth = dataloader.next_element[3]
        intrinsics = dataloader.next_element[4]

        # Build computation graph
        model = FusionNetModel(
            image0=image0,
            image1=image1,
            image2=image2,
            input_depth=input_depth,
            intrinsics=intrinsics,
            is_training=True,
            network_type=network_type,
            image_filter_pct=image_filter_pct,
            depth_filter_pct=depth_filter_pct,
            activation_func=activation_func,
            min_predict_depth=min_predict_depth,
            max_predict_depth=max_predict_depth,
            min_scale_depth=min_scale_depth,
            max_scale_depth=max_scale_depth,
            min_residual_depth=min_residual_depth,
            max_residual_depth=max_residual_depth,
            validity_map_color=validity_map_color,
            w_color=w_color,
            w_structure=w_structure,
            w_sparse_depth=w_sparse_depth,
            w_smoothness=w_smoothness,
            w_prior_depth=w_prior_depth,
            residual_threshold_prior_depth=residual_threshold_prior_depth,
            rotation_param=rotation_param)

        loss = model.loss
        gradients = optimizer.compute_gradients(loss)
        gradients = optimizer.apply_gradients(gradients, global_step=global_step)

        model_summary = tf.summary.merge_all()

        # Count trainable parameters
        n_parameter = 0
        for variable in tf.trainable_variables():
            n_parameter += np.array(variable.get_shape().as_list()).prod()

        # Log settings
        log('Dataloader settings:', log_path)
        log('n_batch=%d  n_height=%d  n_width=%d' %
            (n_batch, n_height, n_width), log_path)
        log('crop_type=%s' %
            (crop_type), log_path)
        log('random_horizontal_crop=%s  random_vertical_crop=%s' %
            (augmentation_random_horizontal_crop, augmentation_random_vertical_crop), log_path)
        log('', log_path)

        log('Network settings:', log_path)
        log('network_type=%s  n_parameter=%d' %
            (network_type, n_parameter), log_path)
        log('image_filter_pct=%.2f  depth_filter_pct=%.2f' %
            (image_filter_pct, depth_filter_pct), log_path)
        log('activation_func=%s' %
            (activation_func), log_path)
        log('', log_path)

        log('Depth prediction settings:', log_path)
        log('min_predict_depth=%.2f  max_predict_depth=%.2f' %
            (min_predict_depth, max_predict_depth), log_path)
        log('min_scale_depth=%.2f  max_scale_depth=%.2f' %
            (min_scale_depth, max_scale_depth), log_path)
        log('min_residual_depth=%.2f  max_residual_depth=%.2f' %
            (min_residual_depth, max_residual_depth), log_path)
        log('', log_path)

        log('Training settings:', log_path)
        log('n_sample=%d  n_epoch=%d  n_step=%d' %
            (n_train_sample, n_epoch, n_train_step), log_path)
        log('learning_schedule=[%s, %d (%d)]' %
            (', '.join('{} ({}) : {:.1E}'.format(s, v, r)
            for s, v, r in zip(
                [0] + learning_schedule, [0] + learning_schedule_steps, learning_rates)),
            n_epoch,
            n_train_step), log_path)
        log('validity_map_color=%s' %
            (validity_map_color), log_path)
        log('w_color=%.2f  w_structure=%.2f  w_sparse_depth=%.2f' %
            (w_color, w_structure, w_sparse_depth), log_path)
        log('w_smoothness=%.3f  w_prior_depth=%.2f' %
            (w_smoothness, w_prior_depth), log_path)
        log('residual_threshold_prior_depth=%.2f' %
            (residual_threshold_prior_depth), log_path)
        log('rotation_param=%s' %
            (rotation_param), log_path)
        log('', log_path)

        log('Depth evaluation settings:', log_path)
        log('min_evaluate_depth=%.2f  max_evaluate_depth=%.2f' %
            (min_evaluate_depth, max_evaluate_depth), log_path)
        log('', log_path)

        log('Checkpoint settings:', log_path)
        log('checkpoint_path=%s' %
            (checkpoint_path), log_path)
        log('restore_path=%s' %
            ('None' if restore_path == '' else restore_path), log_path)
        log('', log_path)

        # Initialize Tensorflow session
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        session = tf.Session(config=config)

        # Initialize saver for storing and restoring checkpoints
        train_summary_writer = tf.summary.FileWriter(event_path + '-train', session.graph)
        val_summary_writer = tf.summary.FileWriter(event_path + '-val')
        train_saver = tf.train.Saver(max_to_keep=50)

        # Initialize all variables
        session.run(tf.global_variables_initializer())
        session.run(tf.local_variables_initializer())

        # If given, load the weights from the restore path
        if restore_path != '':
            import tensorflow.contrib.slim as slim

            vars_to_restore_fusionnet = tf.get_collection(
                tf.GraphKeys.GLOBAL_VARIABLES,
                scope=network_type)
            init_assign_op_fusionnet, init_feed_dict_fusionnet = slim.assign_from_checkpoint(
                restore_path,
                vars_to_restore_fusionnet,
                ignore_missing_vars=True)
            session.run(init_assign_op_fusionnet, init_feed_dict_fusionnet)

            vars_to_restore_posenet = tf.get_collection(
                tf.GraphKeys.GLOBAL_VARIABLES,
                scope='posenet')
            init_assign_op_posenet, init_feed_dict_posenet = slim.assign_from_checkpoint(
                restore_path,
                vars_to_restore_posenet,
                ignore_missing_vars=True)
            session.run(init_assign_op_posenet, init_feed_dict_posenet)

        # Begin training
        log('Begin training...', log_path)
        start_step = global_step.eval(session=session)
        time_start = time.time()
        train_step = start_step
        step = 0

        do_center_crop = True if crop_type == 'center' else False
        do_bottom_crop = True if crop_type == 'bottom' else False

        # Shuffle data for current epoch
        train_image_paths_epoch, \
            train_input_depth_paths_epoch, \
            train_sparse_depth_paths_epoch, \
            train_intrinsics_paths_epoch = data_utils.make_epoch(
                input_arr=[
                    train_image_paths,
                    train_input_depth_paths,
                    train_sparse_depth_paths,
                    train_intrinsics_paths],
                n_batch=n_batch)

        # Feed input paths into dataloader for training
        dataloader.initialize(
            session,
            image_paths=train_image_paths_epoch,
            input_depth_paths=train_input_depth_paths_epoch,
            sparse_depth_paths=train_sparse_depth_paths_epoch,
            intrinsics_paths=train_intrinsics_paths_epoch,
            do_center_crop=do_center_crop,
            do_bottom_crop=do_bottom_crop,
            random_horizontal_crop=augmentation_random_horizontal_crop,
            random_vertical_crop=augmentation_random_vertical_crop)

        while train_step < n_train_step:
            try:
                if train_step % n_summary == 0:
                    # Compute loss and training summary
                    _, loss_value, train_summary = session.run([gradients, loss, model_summary])

                    # Write training summary
                    train_summary_writer.add_summary(train_summary, global_step=train_step)
                else:
                    # Compute loss
                    _, loss_value = session.run([gradients, loss])

                if train_step and (train_step % n_checkpoint) == 0:
                    time_elapse = (time.time() - time_start) / 3600 * train_step / (train_step - start_step + 1)
                    time_remain = (n_train_step / train_step - 1) * time_elapse

                    checkpoint_log = \
                        'batch: {:>6}/{:>6}  time elapsed: {:.2f}h  time left: {:.2f}h \n' + \
                        'loss: {:.5f}'

                    log(checkpoint_log.format(
                        train_step,
                        n_train_step,
                        time_elapse,
                        time_remain,
                        loss_value), log_path)

                    # Feed input paths into dataloader for validation
                    dataloader.initialize(
                        session,
                        image_paths=val_image_paths,
                        input_depth_paths=val_input_depth_paths,
                        sparse_depth_paths=val_sparse_depth_paths,
                        intrinsics_paths=train_intrinsics_paths[0:len(val_image_paths)],
                        do_center_crop=do_center_crop,
                        do_bottom_crop=do_bottom_crop,
                        random_horizontal_crop=False,
                        random_vertical_crop=False)

                    # Run model on validation samples
                    val_output_depths = run(
                        model,
                        session,
                        n_sample=n_val_sample,
                        summary=model_summary,
                        summary_writer=val_summary_writer,
                        step=train_step,
                        verbose=False)

                    eval_utils.evaluate(
                        val_output_depths,
                        val_ground_truths,
                        train_step,
                        log_path=log_path,
                        min_evaluate_depth=min_evaluate_depth,
                        max_evaluate_depth=max_evaluate_depth)

                    # Switch back to training
                    current_sample = n_batch * (step + 1)

                    dataloader.initialize(
                        session,
                        image_paths=train_image_paths_epoch[current_sample:],
                        input_depth_paths=train_input_depth_paths_epoch[current_sample:],
                        sparse_depth_paths=train_sparse_depth_paths_epoch[current_sample:],
                        intrinsics_paths=train_intrinsics_paths_epoch[current_sample:],
                        do_center_crop=do_center_crop,
                        do_bottom_crop=do_bottom_crop,
                        random_horizontal_crop=augmentation_random_horizontal_crop,
                        random_vertical_crop=augmentation_random_vertical_crop)

                    train_saver.save(session, model_path, global_step=train_step)

                train_step += 1
                step += 1
            except tf.errors.OutOfRangeError:
                step = 0

                # Shuffle data for next epoch
                train_image_paths_epoch, \
                    train_input_depth_paths_epoch, \
                    train_sparse_depth_paths_epoch, \
                    train_intrinsics_paths_epoch, = data_utils.make_epoch(
                        input_arr=[
                            train_image_paths,
                            train_input_depth_paths,
                            train_sparse_depth_paths,
                            train_intrinsics_paths],
                        n_batch=n_batch)

                # Feed input paths into dataloader for training
                dataloader.initialize(
                    session,
                    image_paths=train_image_paths_epoch,
                    input_depth_paths=train_input_depth_paths_epoch,
                    sparse_depth_paths=train_sparse_depth_paths_epoch,
                    intrinsics_paths=train_intrinsics_paths_epoch,
                    do_center_crop=do_center_crop,
                    do_bottom_crop=do_bottom_crop,
                    random_horizontal_crop=augmentation_random_horizontal_crop,
                    random_vertical_crop=augmentation_random_vertical_crop)

        train_saver.save(session, model_path, global_step=n_train_step)

def run(model, session, n_sample, summary=None, summary_writer=None, step=-1, verbose=False):

    output_depths = []
    n_processed = 0

    while True:
        try:
            if summary is not None and summary_writer is not None:
                # Run model and summary
                output_depth, model_summary = session.run([model.predict, summary])

                # Write results to summary
                summary_writer.add_summary(model_summary, global_step=step)
            else:
                # Run model
                output_depth = session.run(model.predict)

            output_depths.append(output_depth)

            if verbose:
                n_processed += output_depth.shape[0]
                print('Processed {}/{} examples'.format(n_processed, n_sample), end='\r')

        except tf.errors.OutOfRangeError:
            break

    # Drop samples used for padding batch
    output_depths = np.concatenate(output_depths, axis=0)
    output_depths = output_depths[0:n_sample, ...]

    return output_depths
