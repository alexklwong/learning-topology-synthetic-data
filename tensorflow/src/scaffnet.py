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
from scaffnet_dataloader import ScaffNetDataloader
from scaffnet_model import ScaffNetModel
from log_utils import log


def train(train_sparse_depth_path,
          train_ground_truth_path,
          # Validation data
          val_sparse_depth_path=None,
          val_ground_truth_path=None,
          # Dataloader settings
          n_batch=settings.N_BATCH,
          n_height=settings.N_HEIGHT,
          n_width=settings.N_WIDTH,
          min_dataset_depth=settings.MIN_DATASET_DEPTH,
          max_dataset_depth=settings.MAX_DATASET_DEPTH,
          crop_type=settings.CROP_TYPE,
          augmentation_random_horizontal_crop=False,
          augmentation_random_vertical_crop=False,
          augmentation_random_horizontal_flip=False,
          # Network architecture
          network_type=settings.NETWORK_TYPE_SCAFFNET,
          activation_func=settings.ACTIVATION_FUNC,
          n_filter_output=settings.N_FILTER_OUTPUT_SCAFFNET,
          # Spatial pyramid pooling
          pool_kernel_sizes_spp=settings.POOL_KERNEL_SIZES_SPP,
          n_convolution_spp=settings.N_CONVOLUTION_SPP,
          n_filter_spp=settings.N_FILTER_SPP,
          # Depth prediction settings
          min_predict_depth=settings.MIN_PREDICT_DEPTH,
          max_predict_depth=settings.MAX_PREDICT_DEPTH,
          # Training settings
          learning_rates=settings.LEARNING_RATES,
          learning_schedule=settings.LEARNING_SCHEDULE,
          n_epoch=settings.N_EPOCH,
          loss_func=settings.LOSS_FUNC_SCAFFNET,
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

    # Load sparse depth, validity map and ground truth paths from file for training
    train_sparse_depth_paths = data_utils.read_paths(train_sparse_depth_path)
    train_ground_truth_paths = data_utils.read_paths(train_ground_truth_path)

    n_train_sample = len(train_sparse_depth_paths)

    assert n_train_sample == len(train_ground_truth_paths)

    n_train_step = n_epoch * np.ceil(n_train_sample / n_batch).astype(np.int32)

    # Load sparse depth, validity map and ground truth paths from file for validation
    val_sparse_depth_paths = data_utils.read_paths(val_sparse_depth_path)
    val_ground_truth_paths = data_utils.read_paths(val_ground_truth_path)

    n_val_sample = len(val_sparse_depth_paths)

    assert n_val_sample == len(val_ground_truth_paths)

    # Pad validation paths based on batch size
    val_sparse_depth_paths = data_utils.pad_batch(val_sparse_depth_paths, n_batch)

    # Load validation ground truth and do center crop
    val_ground_truths = []
    for idx in range(len(val_ground_truth_paths)):
        # Load ground truth and validity map
        ground_truth, validity_map = \
            data_utils.load_depth_with_validity_map(val_ground_truth_paths[idx])

        # Get crop start and end positions
        if crop_type == 'center':
            start_height = int(float(ground_truth.shape[0] - n_height))
        elif crop_type == 'bottom':
            start_height = ground_truth.shape[0] - n_height
        else:
            start_height = 0

        end_height = n_height + start_height

        start_width = int(float(ground_truth.shape[1] - n_width) / 2.0)
        end_width = n_width + start_width

        # Concatenate ground truth and validity map together
        ground_truth = np.concatenate([
            np.expand_dims(ground_truth, axis=-1),
            np.expand_dims(validity_map, axis=-1)], axis=-1)

        # Crop ground truth
        val_ground_truths.append(
            ground_truth[start_height:end_height, start_width:end_width, :])

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
        dataloader = ScaffNetDataloader(
            shape=[n_batch, n_height, n_width, 2],
            name='scaffnet_dataloader',
            is_training=True,
            n_thread=n_thread,
            prefetch_size=(2 * n_thread))

        # Fetch the input from dataloader
        input_depth = dataloader.next_element[0]
        ground_truth = dataloader.next_element[1]

        # Build computation graph
        model = ScaffNetModel(
            input_depth,
            ground_truth,
            is_training=True,
            network_type=network_type,
            activation_func=activation_func,
            n_filter_output=n_filter_output,
            pool_kernel_sizes_spp=pool_kernel_sizes_spp,
            n_convolution_spp=n_convolution_spp,
            n_filter_spp=n_filter_spp,
            min_dataset_depth=min_dataset_depth,
            max_dataset_depth=max_dataset_depth,
            min_predict_depth=min_predict_depth,
            max_predict_depth=max_predict_depth,
            loss_func=loss_func)

        # Compute loss and gradients
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
        log('min_dataset_depth=%.2f  max_dataset_depth=%.2f' %
            (min_dataset_depth, max_dataset_depth), log_path)
        log('crop_type=%s' %
            (crop_type), log_path)
        log('random_horizontal_crop=%s  random_vertical_crop=%s' %
            (augmentation_random_horizontal_crop, augmentation_random_vertical_crop), log_path)
        log('random_horizontal_flip=%s' %
            (augmentation_random_horizontal_flip), log_path)
        log('', log_path)

        log('Network settings:', log_path)
        log('network_type=%s  n_parameter=%d' %
            (network_type, n_parameter), log_path)
        log('activation_func=%s' %
            (activation_func), log_path)
        log('n_filter_output=%s' %
            (str(n_filter_output) if n_filter_output > 0 else 'upsample'), log_path)
        log('', log_path)

        log('Spatial pyramid pooling settings:', log_path)
        log('pool_kernel_sizes_spp=[%s]' %
            (', '.join([str(i) for i in pool_kernel_sizes_spp])),
            log_path)
        log('n_convolution_spp=%d  n_filter_spp=%d' %
            (n_convolution_spp, n_filter_spp), log_path)
        log('', log_path)

        log('Depth prediction settings:', log_path)
        log('min_predict_depth=%.2f  max_predict_depth=%.2f' %
            (min_predict_depth, max_predict_depth), log_path)
        log('', log_path)

        log('Training settings:', log_path)
        log('n_sample=%d  n_epoch=%d  n_step=%d' %
            (n_train_sample, n_epoch, n_train_step), log_path)
        log('loss_func=%s' %
            (loss_func), log_path)
        log('learning_schedule=[%s, %d (%d)]' %
            (', '.join('{} ({}) : {:.1E}'.format(s, v, r)
            for s, v, r in zip(
                [0] + learning_schedule, [0] + learning_schedule_steps, learning_rates)),
            n_epoch,
            n_train_step), log_path)
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
            train_saver.restore(session, restore_path)

        # Begin training
        log('Begin training...', log_path)
        start_step = global_step.eval(session=session)
        time_start = time.time()
        train_step = start_step
        step = 0

        do_center_crop = True if crop_type == 'center' else False
        do_bottom_crop = True if crop_type == 'bottom' else False

        # Shuffle data for current epoch
        train_sparse_depth_paths_epoch, \
            train_ground_truth_paths_epoch = data_utils.make_epoch(
                input_arr=[train_sparse_depth_paths, train_ground_truth_paths],
                n_batch=n_batch)

        # Feed input paths into dataloader for training
        dataloader.initialize(
            session,
            sparse_depth_paths=train_sparse_depth_paths_epoch,
            ground_truth_paths=train_ground_truth_paths_epoch,
            do_center_crop=do_center_crop,
            do_bottom_crop=do_bottom_crop,
            random_horizontal_crop=augmentation_random_horizontal_crop,
            random_vertical_crop=augmentation_random_vertical_crop,
            random_horizontal_flip=augmentation_random_horizontal_flip)

        while train_step < n_train_step:
            try:
                if train_step % n_summary == 0:
                    # Compute loss and training summary
                    _, loss_value, train_summary = session.run([gradients, loss, model_summary])

                    # Write training results to summary
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
                        sparse_depth_paths=val_sparse_depth_paths,
                        ground_truth_paths=val_ground_truth_paths,
                        do_center_crop=do_center_crop,
                        do_bottom_crop=do_bottom_crop,
                        random_horizontal_crop=False,
                        random_vertical_crop=False,
                        random_horizontal_flip=False)

                    # Run model on validation samples
                    val_output_depths = run(
                        model,
                        session,
                        n_sample=n_val_sample,
                        summary=model_summary,
                        summary_writer=val_summary_writer,
                        step=train_step,
                        verbose=False)

                    # Run validation metrics
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
                        sparse_depth_paths=train_sparse_depth_paths_epoch[current_sample:],
                        ground_truth_paths=train_ground_truth_paths_epoch[current_sample:],
                        do_center_crop=do_center_crop,
                        do_bottom_crop=do_bottom_crop,
                        random_horizontal_crop=augmentation_random_horizontal_crop,
                        random_vertical_crop=augmentation_random_vertical_crop,
                        random_horizontal_flip=augmentation_random_horizontal_flip)

                    train_saver.save(session, model_path, global_step=train_step)

                train_step += 1
                step += 1
            except tf.errors.OutOfRangeError:
                step = 0

                # Shuffle data for next epoch
                train_sparse_depth_paths_epoch, \
                    train_ground_truth_paths_epoch = data_utils.make_epoch(
                        input_arr=[train_sparse_depth_paths, train_ground_truth_paths],
                        n_batch=n_batch)

                # Feed input paths into dataloader for training
                dataloader.initialize(
                    session,
                    sparse_depth_paths=train_sparse_depth_paths_epoch,
                    ground_truth_paths=train_ground_truth_paths_epoch,
                    do_center_crop=do_center_crop,
                    do_bottom_crop=do_bottom_crop,
                    random_horizontal_crop=augmentation_random_horizontal_crop,
                    random_vertical_crop=augmentation_random_vertical_crop,
                    random_horizontal_flip=augmentation_random_horizontal_flip)

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
