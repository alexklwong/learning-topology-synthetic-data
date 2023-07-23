import os, time
import numpy as np
from PIL import Image
import torch
from torch.utils.tensorboard import SummaryWriter
import datasets, data_utils, eval_utils
from log_utils import log
from fusionnet_model import FusionNetModel
from scaffnet_model import ScaffNetModel
from posenet_model import PoseNetModel
from transforms import Transforms
from net_utils import OutlierRemoval
from scaffnet_main import log_scaffnet_settings


def train(train_images_path,
          train_sparse_depth_path,
          train_intrinsics_path,
          val_image_path,
          val_sparse_depth_path,
          val_ground_truth_path,
          # Batch settings
          n_batch,
          n_height,
          n_width,
          # Input settings
          normalized_image_range,
          outlier_removal_kernel_size,
          outlier_removal_threshold,
          # Spatial pyramid pool settings
          max_pool_sizes_spatial_pyramid_pool,
          n_convolution_spatial_pyramid_pool,
          n_filter_spatial_pyramid_pool,
          # ScaffNet settings
          encoder_type_scaffnet,
          n_filters_encoder_scaffnet,
          decoder_type_scaffnet,
          n_filters_decoder_scaffnet,
          min_predict_depth_scaffnet,
          max_predict_depth_scaffnet,
          # FusionNet network settings
          encoder_type_fusionnet,
          n_filters_encoder_image_fusionnet,
          n_filters_encoder_depth_fusionnet,
          decoder_type_fusionnet,
          n_filters_decoder_fusionnet,
          scale_match_method_fusionnet,
          scale_match_kernel_size_fusionnet,
          min_predict_depth_fusionnet,
          max_predict_depth_fusionnet,
          min_multiplier_depth_fusionnet,
          max_multiplier_depth_fusionnet,
          min_residual_depth_fusionnet,
          max_residual_depth_fusionnet,
          # Weight settings
          weight_initializer,
          activation_func,
          # Training settings
          learning_rates,
          learning_schedule,
          freeze_scaffnet_modules,
          # Augmentation settings
          augmentation_probabilities,
          augmentation_schedule,
          augmentation_random_crop_type,
          augmentation_random_crop_to_shape,
          augmentation_random_flip_type,
          augmentation_random_remove_points,
          augmentation_random_noise_type,
          augmentation_random_noise_spread,
          augmentation_random_brightness,
          augmentation_random_contrast,
          augmentation_random_saturation,
          # Loss function settings
          w_color,
          w_structure,
          w_sparse_depth,
          w_smoothness,
          w_prior_depth,
          threshold_prior_depth,
          w_weight_decay_depth,
          w_weight_decay_pose,
          # Evaluation settings
          min_evaluate_depth,
          max_evaluate_depth,
          # Checkpoint settings
          n_step_per_summary,
          n_image_per_summary,
          n_step_per_checkpoint,
          checkpoint_path,
          start_step_validation,
          scaffnet_model_restore_path,
          fusionnet_model_restore_path,
          posenet_model_restore_path,
          # Hardware settings
          device='cuda',
          n_thread=8):

    # Select device to run on
    if device == 'cuda' or device == 'gpu':
        device = torch.device('cuda')
    elif device == 'cpu':
        device = torch.device('cpu')
    else:
        raise ValueError('Unsupported device: {}'.format(device))

    # Set up checkpoint and event paths
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)

    depth_model_checkpoint_path = os.path.join(checkpoint_path, 'fusionnet-{}.pth')
    pose_model_checkpoint_path = os.path.join(checkpoint_path, 'posenet-{}.pth')
    log_path = os.path.join(checkpoint_path, 'results.txt')
    event_path = os.path.join(checkpoint_path, 'events')

    best_results = {
        'step': -1,
        'mae': np.infty,
        'rmse': np.infty,
        'imae': np.infty,
        'irmse': np.infty,
        'mare': np.infty
    }

    '''
    Load input paths and set up dataloaders
    '''
    train_images_paths = data_utils.read_paths(train_images_path)
    train_sparse_depth_paths = data_utils.read_paths(train_sparse_depth_path)

    n_train_sample = len(train_images_paths)

    if train_intrinsics_path is None:
        train_intrinsics_paths = [None] * n_train_sample
    else:
        train_intrinsics_paths = data_utils.read_paths(train_intrinsics_path)

    # Make sure number of paths match number of training sample
    input_paths = [
        train_sparse_depth_paths,
        train_intrinsics_paths,
    ]

    for paths in input_paths:
        assert len(paths) == n_train_sample

    # Set up training dataloader
    n_train_step = \
        learning_schedule[-1] * np.floor(n_train_sample / n_batch).astype(np.int32)

    train_dataloader = torch.utils.data.DataLoader(
        datasets.FusionNetStandaloneTrainingDataset(
            images_paths=train_images_paths,
            sparse_depth_paths=train_sparse_depth_paths,
            intrinsics_paths=train_intrinsics_paths,
            random_crop_shape=(n_height, n_width),
            random_crop_type=augmentation_random_crop_type),
        batch_size=n_batch,
        shuffle=True,
        num_workers=n_thread,
        drop_last=True)

    train_transforms = Transforms(
        normalized_image_range=normalized_image_range,
        random_crop_to_shape=augmentation_random_crop_to_shape,
        random_flip_type=augmentation_random_flip_type,
        random_remove_points=augmentation_random_remove_points,
        random_noise_type=augmentation_random_noise_type,
        random_noise_spread=augmentation_random_noise_spread,
        random_brightness=augmentation_random_brightness,
        random_contrast=augmentation_random_contrast,
        random_saturation=augmentation_random_saturation)

    # Load validation data if it is available
    validation_available = \
        val_image_path is not None and \
        val_sparse_depth_path is not None and \
        val_ground_truth_path is not None

    if validation_available:
        val_image_paths = data_utils.read_paths(val_image_path)
        val_sparse_depth_paths = data_utils.read_paths(val_sparse_depth_path)
        val_ground_truth_paths = data_utils.read_paths(val_ground_truth_path)

        n_val_sample = len(val_image_paths)

        input_paths = [
            val_sparse_depth_paths, val_ground_truth_paths
        ]

        for paths in input_paths:
            assert len(paths) == n_val_sample

        val_dataloader = torch.utils.data.DataLoader(
            datasets.FusionNetStandaloneInferenceDataset(
                image_paths=val_image_paths,
                sparse_depth_paths=val_sparse_depth_paths,
                ground_truth_paths=val_ground_truth_paths,
                load_image_triplets=False),
            batch_size=1,
            shuffle=False,
            num_workers=1,
            drop_last=False)

        val_transforms = Transforms(
            normalized_image_range=normalized_image_range)

    # Initialize outlier removal for sparse depth
    outlier_removal = OutlierRemoval(
        kernel_size=outlier_removal_kernel_size,
        threshold=outlier_removal_threshold)

    # Build ScaffNet
    scaffnet_model = ScaffNetModel(
        max_pool_sizes_spatial_pyramid_pool=max_pool_sizes_spatial_pyramid_pool,
        n_convolution_spatial_pyramid_pool=n_convolution_spatial_pyramid_pool,
        n_filter_spatial_pyramid_pool=n_filter_spatial_pyramid_pool,
        encoder_type=encoder_type_scaffnet,
        n_filters_encoder=n_filters_encoder_scaffnet,
        decoder_type=decoder_type_scaffnet,
        n_filters_decoder=n_filters_decoder_scaffnet,
        n_output_resolution=1,
        weight_initializer=weight_initializer,
        activation_func=activation_func,
        min_predict_depth=min_predict_depth_scaffnet,
        max_predict_depth=max_predict_depth_scaffnet,
        device=device)

    scaffnet_model.restore_model(scaffnet_model_restore_path)

    scaffnet_model.eval()

    # Freeze a portion of the network
    if freeze_scaffnet_modules is not None:
        scaffnet_model.freeze(freeze_scaffnet_modules)

    parameters_scaffnet_model = scaffnet_model.parameters()

    # Build FusionNet
    fusionnet_model = FusionNetModel(
        encoder_type=encoder_type_fusionnet,
        n_filters_encoder_image=n_filters_encoder_image_fusionnet,
        n_filters_encoder_depth=n_filters_encoder_depth_fusionnet,
        decoder_type=decoder_type_fusionnet,
        n_filters_decoder=n_filters_decoder_fusionnet,
        scale_match_method=scale_match_method_fusionnet,
        scale_match_kernel_size=scale_match_kernel_size_fusionnet,
        min_predict_depth=min_predict_depth_fusionnet,
        max_predict_depth=max_predict_depth_fusionnet,
        min_multiplier_depth=min_multiplier_depth_fusionnet,
        max_multiplier_depth=max_multiplier_depth_fusionnet,
        min_residual_depth=min_residual_depth_fusionnet,
        max_residual_depth=max_residual_depth_fusionnet,
        weight_initializer=weight_initializer,
        activation_func=activation_func,
        device=device)

    fusionnet_model.train()
    parameters_fusionnet_model = fusionnet_model.parameters()

    # Bulid PoseNet (only needed for unsupervised training) network
    posenet_model = PoseNetModel(
        encoder_type='posenet',
        rotation_parameterization='axis',
        weight_initializer=weight_initializer,
        activation_func='relu',
        device=device)

    posenet_model.train()
    parameters_posenet_model = posenet_model.parameters()

    # Set up tensorboard summary writers
    train_summary_writer = SummaryWriter(event_path + '-train')
    val_summary_writer = SummaryWriter(event_path + '-val')

    '''
    Log input paths
    '''
    log('Training input paths:', log_path)
    train_input_paths = [
        train_images_path,
        train_sparse_depth_path,
        train_intrinsics_path
    ]

    for path in train_input_paths:
        if path is not None:
            log(path, log_path)
    log('', log_path)

    log('Validation input paths:', log_path)
    val_input_paths = [
        val_image_path,
        val_sparse_depth_path,
        val_ground_truth_path
    ]
    for path in val_input_paths:
        log(path, log_path)
    log('', log_path)

    log_input_settings(
        log_path,
        # Batch settings
        n_batch=n_batch,
        n_height=n_height,
        n_width=n_width,
        # Input settings
        normalized_image_range=normalized_image_range,
        outlier_removal_kernel_size=outlier_removal_kernel_size,
        outlier_removal_threshold=outlier_removal_threshold)

    log_scaffnet_settings(
        log_path,
        # Spatial pyramid pool settings
        max_pool_sizes_spatial_pyramid_pool=max_pool_sizes_spatial_pyramid_pool,
        n_convolution_spatial_pyramid_pool=n_convolution_spatial_pyramid_pool,
        n_filter_spatial_pyramid_pool=n_filter_spatial_pyramid_pool,
        # Depth network settings
        encoder_type=encoder_type_scaffnet,
        n_filters_encoder=n_filters_encoder_scaffnet,
        decoder_type=decoder_type_scaffnet,
        n_filters_decoder=n_filters_decoder_scaffnet,
        min_predict_depth=min_predict_depth_scaffnet,
        max_predict_depth=max_predict_depth_scaffnet,
        n_output_resolution=1,
        # Weight settings
        weight_initializer=weight_initializer,
        activation_func=activation_func,
        parameters_model=parameters_scaffnet_model)

    log_fusionnet_settings(
        log_path,
        # Depth network settings
        encoder_type=encoder_type_fusionnet,
        n_filters_encoder_image=n_filters_encoder_image_fusionnet,
        n_filters_encoder_depth=n_filters_encoder_depth_fusionnet,
        decoder_type=decoder_type_fusionnet,
        n_filters_decoder=n_filters_decoder_fusionnet,
        scale_match_method=scale_match_method_fusionnet,
        scale_match_kernel_size=scale_match_kernel_size_fusionnet,
        min_predict_depth=min_predict_depth_fusionnet,
        max_predict_depth=max_predict_depth_fusionnet,
        min_multiplier_depth=min_multiplier_depth_fusionnet,
        max_multiplier_depth=max_multiplier_depth_fusionnet,
        min_residual_depth=min_residual_depth_fusionnet,
        max_residual_depth=max_residual_depth_fusionnet,
        # Weight settings
        weight_initializer=weight_initializer,
        activation_func=activation_func,
        parameters_depth_model=parameters_fusionnet_model,
        parameters_pose_model=parameters_posenet_model)

    log_training_settings(
        log_path,
        # Training settings
        n_batch=n_batch,
        n_train_sample=n_train_sample,
        n_train_step=n_train_step,
        learning_rates=learning_rates,
        learning_schedule=learning_schedule,
        freeze_scaffnet_modules=freeze_scaffnet_modules,
        # Augmentation settings
        augmentation_probabilities=augmentation_probabilities,
        augmentation_schedule=augmentation_schedule,
        augmentation_random_crop_type=augmentation_random_crop_type,
        augmentation_random_crop_to_shape=augmentation_random_crop_to_shape,
        augmentation_random_flip_type=augmentation_random_flip_type,
        augmentation_random_remove_points=augmentation_random_remove_points,
        augmentation_random_noise_type=augmentation_random_noise_type,
        augmentation_random_noise_spread=augmentation_random_noise_spread,
        augmentation_random_brightness=augmentation_random_brightness,
        augmentation_random_contrast=augmentation_random_contrast,
        augmentation_random_saturation=augmentation_random_saturation)

    log_loss_func_settings(
        log_path,
        # Loss function settings
        w_color=w_color,
        w_structure=w_structure,
        w_sparse_depth=w_sparse_depth,
        w_smoothness=w_smoothness,
        w_prior_depth=w_prior_depth,
        threshold_prior_depth=threshold_prior_depth,
        w_weight_decay_depth=w_weight_decay_depth,
        w_weight_decay_pose=w_weight_decay_pose)

    log_evaluation_settings(
        log_path,
        min_evaluate_depth=min_evaluate_depth,
        max_evaluate_depth=max_evaluate_depth)

    log_system_settings(
        log_path,
        # Checkpoint settings
        checkpoint_path=checkpoint_path,
        n_step_per_checkpoint=n_step_per_checkpoint,
        summary_event_path=event_path,
        n_step_per_summary=n_step_per_summary,
        n_image_per_summary=n_image_per_summary,
        start_step_validation=start_step_validation,
        depth_model_restore_path=fusionnet_model_restore_path,
        pose_model_restore_path=posenet_model_restore_path,
        # Hardware settings
        device=device,
        n_thread=n_thread)

    '''
    Train model
    '''
    # Initialize optimizer with starting learning rate
    learning_schedule_pos = 0
    learning_rate = learning_rates[0]

    augmentation_schedule_pos = 0
    augmentation_probability = augmentation_probabilities[0]

    optimizer_fusionnet = torch.optim.Adam([
        {
            'params' : parameters_scaffnet_model,
            'weight_decay' : w_weight_decay_depth
        }, {
            'params' :  parameters_fusionnet_model,
            'weight_decay' : w_weight_decay_depth
        }],
        lr=learning_rate)

    optimizer_posenet = torch.optim.Adam([
        {
            'params' : parameters_posenet_model,
            'weight_decay' : w_weight_decay_pose
        }],
        lr=learning_rate)

    # Start training
    start_step = 0

    if fusionnet_model_restore_path is not None:
        start_step, optimizer_fusionnet = fusionnet_model.restore_model(
            fusionnet_model_restore_path,
            optimizer=optimizer_fusionnet,
            scaffnet_model=scaffnet_model if scaffnet_model_restore_path is None else None)

        for g in optimizer_fusionnet.param_groups:
            g['lr'] = learning_rate

    if posenet_model_restore_path is not None:
        _, optimizer_posenet = posenet_model.restore_model(
            posenet_model_restore_path,
            optimizer=optimizer_posenet)

        for g in optimizer_posenet.param_groups:
            g['lr'] = learning_rate

    train_step = start_step

    time_start = time.time()

    # Split along batch across multiple GPUs
    if torch.cuda.device_count() > 1:
        scaffnet_model.data_parallel()
        fusionnet_model.data_parallel()

    log('Begin training...', log_path)
    for epoch in range(1, learning_schedule[-1] + 1):

        # Set learning rate schedule
        if epoch > learning_schedule[learning_schedule_pos]:
            learning_schedule_pos = learning_schedule_pos + 1
            learning_rate = learning_rates[learning_schedule_pos]

            # Update optimizer learning rates
            for g in optimizer_fusionnet.param_groups:
                g['lr'] = learning_rate

            for g in optimizer_posenet.param_groups:
                g['lr'] = learning_rate

        # Set augmentation schedule
        if -1 not in augmentation_schedule and epoch > augmentation_schedule[augmentation_schedule_pos]:
            augmentation_schedule_pos = augmentation_schedule_pos + 1
            augmentation_probability = augmentation_probabilities[augmentation_schedule_pos]

        for inputs in train_dataloader:

            train_step = train_step + 1

            # Fetch data
            inputs = [
                in_.to(device) for in_ in inputs
            ]

            image0, \
                image1, \
                image2, \
                sparse_depth0, \
                intrinsics = inputs

            with torch.no_grad():
                # Forward through ScaffNet
                input_depth0 = scaffnet_model.forward(sparse_depth0)

                if 'uncertainty' in scaffnet_model.decoder_type:
                    input_depth0 = input_depth0[:, 0:1, :, :]

            # Validity map is where sparse depth is available
            validity_map0 = torch.where(
                sparse_depth0 > 0,
                torch.ones_like(sparse_depth0),
                sparse_depth0)

            # Remove outlier points and update sparse depth and validity map
            filtered_sparse_depth0, \
                filtered_validity_map0 = outlier_removal.remove_outliers(
                    sparse_depth=sparse_depth0,
                    validity_map=validity_map0)

            # Transforms
            [image0, image1, image2], \
                [filtered_sparse_depth0], \
                [input_depth0, filtered_validity_map0], \
                [intrinsics] = train_transforms.transform(
                    images_arr=[image0, image1, image2],
                    range_maps_arr=[filtered_sparse_depth0],
                    validity_maps_arr=[input_depth0, filtered_validity_map0],
                    intrinsics_arr=[intrinsics],
                    random_transform_probability=augmentation_probability)

            # Forward through FusionNet
            output_depth0 = fusionnet_model.forward(
                image=image0,
                input_depth=input_depth0,
                sparse_depth=filtered_sparse_depth0)

            # Forward through PoseNet
            pose0to1 = posenet_model.forward(image0, image1)
            pose0to2 = posenet_model.forward(image0, image2)

            # Compute loss function
            loss, loss_info = fusionnet_model.compute_loss_unsupervised(
                output_depth0=output_depth0,
                sparse_depth0=filtered_sparse_depth0,
                validity_map0=filtered_validity_map0,
                input_depth0=input_depth0,
                image0=image0,
                image1=image1,
                image2=image2,
                pose0to1=pose0to1,
                pose0to2=pose0to2,
                intrinsics=intrinsics,
                w_color=w_color,
                w_structure=w_structure,
                w_sparse_depth=w_sparse_depth,
                w_smoothness=w_smoothness,
                w_prior_depth=w_prior_depth,
                threshold_prior_depth=threshold_prior_depth)

            # Compute gradient and backpropagate
            optimizer_fusionnet.zero_grad()
            optimizer_posenet.zero_grad()
            loss.backward()
            optimizer_fusionnet.step()
            optimizer_posenet.step()

            if (train_step % n_step_per_summary) == 0:
                image1to0 = loss_info.pop('image1to0')
                image2to0 = loss_info.pop('image2to0')

                fusionnet_model.log_summary(
                    summary_writer=train_summary_writer,
                    tag='train',
                    step=train_step,
                    image0=image0,
                    image1to0=image1to0.detach().clone(),
                    image2to0=image2to0.detach().clone(),
                    output_depth0=output_depth0.detach().clone(),
                    sparse_depth0=filtered_sparse_depth0,
                    validity_map0=filtered_validity_map0,
                    input_depth0=input_depth0.detach().clone(),
                    pose0to1=pose0to1,
                    pose0to2=pose0to2,
                    scalars=loss_info,
                    n_image_per_summary=min(n_batch, n_step_per_summary))

            # Log results and save checkpoints
            if (train_step % n_step_per_checkpoint) == 0:
                time_elapse = (time.time() - time_start) / 3600
                time_remain = (n_train_step - train_step + start_step) * time_elapse / (train_step - start_step)

                log('Step={:6}/{}  Loss={:.5f}  Time Elapsed={:.2f}h  Time Remaining={:.2f}h'.format(
                    train_step, n_train_step + start_step, loss.item(), time_elapse, time_remain), log_path)

                if train_step >= start_step_validation and validation_available:
                    # Switch to validation mode
                    fusionnet_model.eval()

                    with torch.no_grad():
                        # Perform validation
                        best_results = validate(
                            scaffnet_model=scaffnet_model,
                            fusionnet_model=fusionnet_model,
                            dataloader=val_dataloader,
                            transforms=val_transforms,
                            outlier_removal=outlier_removal,
                            step=train_step,
                            best_results=best_results,
                            min_evaluate_depth=min_evaluate_depth,
                            max_evaluate_depth=max_evaluate_depth,
                            device=device,
                            summary_writer=val_summary_writer,
                            n_image_per_summary=n_image_per_summary,
                            log_path=log_path)

                    # Switch back to training
                    fusionnet_model.train()

                fusionnet_model.save_model(
                    depth_model_checkpoint_path.format(train_step),
                    train_step,
                    optimizer_fusionnet,
                    scaffnet_model)

                posenet_model.save_model(
                    pose_model_checkpoint_path.format(train_step),
                    train_step,
                    optimizer_posenet)

    log('Step={:6}/{}  Loss={:.5f}  Time Elapsed={:.2f}h  Time Remaining={:.2f}h'.format(
        train_step, n_train_step + start_step, loss.item(), time_elapse, time_remain), log_path)

    # Switch to validation mode
    fusionnet_model.eval()

    with torch.no_grad():
        # Perform validation
        best_results = validate(
            scaffnet_model=scaffnet_model,
            fusionnet_model=fusionnet_model,
            dataloader=val_dataloader,
            transforms=val_transforms,
            outlier_removal=outlier_removal,
            step=train_step,
            best_results=best_results,
            min_evaluate_depth=min_evaluate_depth,
            max_evaluate_depth=max_evaluate_depth,
            device=device,
            summary_writer=val_summary_writer,
            n_image_per_summary=n_image_per_summary,
            log_path=log_path)

    fusionnet_model.save_model(
        depth_model_checkpoint_path.format(train_step),
        train_step,
        optimizer_fusionnet,
        scaffnet_model)

    posenet_model.save_model(
        pose_model_checkpoint_path.format(train_step),
        train_step,
        optimizer_posenet)

def validate(scaffnet_model,
             fusionnet_model,
             dataloader,
             transforms,
             outlier_removal,
             step,
             best_results,
             min_evaluate_depth,
             max_evaluate_depth,
             device,
             summary_writer,
             n_image_per_summary=4,
             n_interval_per_summary=250,
             log_path=None):

    n_sample = len(dataloader)
    mae = np.zeros(n_sample)
    rmse = np.zeros(n_sample)
    imae = np.zeros(n_sample)
    irmse = np.zeros(n_sample)
    mare = np.zeros(n_sample)

    image_summary = []
    output_depth_summary = []
    sparse_depth_summary = []
    input_depth_summary = []
    ground_truth_summary = []

    for idx, inputs in enumerate(dataloader):

        # Move inputs to device
        inputs = [
            in_.to(device) for in_ in inputs
        ]

        image, sparse_depth, ground_truth = inputs

        with torch.no_grad():

            # Forward through ScaffNet
            input_depth = scaffnet_model.forward(sparse_depth)

            if 'uncertainty' in scaffnet_model.decoder_type:
                input_depth = input_depth[:, 0:1, :, :]

            # Validity map is where sparse depth is available
            validity_map = torch.where(
                sparse_depth > 0,
                torch.ones_like(sparse_depth),
                sparse_depth)

            # Remove outlier points and update sparse depth and validity map
            filtered_sparse_depth, _ = outlier_removal.remove_outliers(
                sparse_depth=sparse_depth,
                validity_map=validity_map)

            [image] = transforms.transform(
                images_arr=[image],
                random_transform_probability=0.0)

            # Forward through network
            output_depth = fusionnet_model.forward(
                image=image,
                input_depth=input_depth,
                sparse_depth=filtered_sparse_depth)

        if (idx % n_interval_per_summary) == 0 and summary_writer is not None:
            image_summary.append(image)
            output_depth_summary.append(output_depth)
            sparse_depth_summary.append(filtered_sparse_depth)
            input_depth_summary.append(input_depth)
            ground_truth_summary.append(ground_truth)

        # Convert to numpy to validate
        output_depth = np.squeeze(output_depth.cpu().numpy())
        ground_truth = np.squeeze(ground_truth.cpu().numpy())

        # Select valid regions to evaluate
        validity_mask = np.where(ground_truth > 0, 1, 0)
        min_max_mask = np.logical_and(
            ground_truth > min_evaluate_depth,
            ground_truth < max_evaluate_depth)
        mask = np.where(np.logical_and(validity_mask, min_max_mask) > 0)

        output_depth = output_depth[mask]
        ground_truth = ground_truth[mask]

        # Compute validation metrics
        mae[idx] = eval_utils.mean_abs_err(1000.0 * output_depth, 1000.0 * ground_truth)
        rmse[idx] = eval_utils.root_mean_sq_err(1000.0 * output_depth, 1000.0 * ground_truth)
        imae[idx] = eval_utils.inv_mean_abs_err(0.001 * output_depth, 0.001 * ground_truth)
        irmse[idx] = eval_utils.inv_root_mean_sq_err(0.001 * output_depth, 0.001 * ground_truth)
        mare[idx] = eval_utils.mean_abs_rel_err(output_depth, ground_truth)

    # Compute mean metrics
    mae   = np.mean(mae)
    rmse  = np.mean(rmse)
    imae  = np.mean(imae)
    irmse = np.mean(irmse)
    mare = np.mean(mare)

    # Log to tensorboard
    if summary_writer is not None:
        fusionnet_model.log_summary(
            summary_writer=summary_writer,
            tag='eval',
            step=step,
            image0=torch.cat(image_summary, dim=0),
            output_depth0=torch.cat(output_depth_summary, dim=0),
            sparse_depth0=torch.cat(sparse_depth_summary, dim=0),
            input_depth0=torch.cat(input_depth_summary, dim=0),
            ground_truth0=torch.cat(ground_truth_summary, dim=0),
            scalars={'mae' : mae, 'rmse' : rmse, 'imae' : imae, 'irmse': irmse, 'mare': mare},
            n_image_per_summary=n_image_per_summary)

    # Print validation results to console
    log('Validation results:', log_path)
    log('{:>8}  {:>8}  {:>8}  {:>8}  {:>8}  {:>8}'.format(
        'Step', 'MAE', 'RMSE', 'iMAE', 'iRMSE', 'MARE'),
        log_path)
    log('{:8}  {:8.3f}  {:8.3f}  {:8.3f}  {:8.3f}  {:8.3f}'.format(
        step, mae, rmse, imae, irmse, mare),
        log_path)

    n_improve = 0
    if np.round(mae, 2) <= np.round(best_results['mae'], 2):
        n_improve = n_improve + 1
    if np.round(rmse, 2) <= np.round(best_results['rmse'], 2):
        n_improve = n_improve + 1
    if np.round(imae, 2) <= np.round(best_results['imae'], 2):
        n_improve = n_improve + 1
    if np.round(irmse, 2) <= np.round(best_results['irmse'], 2):
        n_improve = n_improve + 1
    if np.round(mare, 4) <= np.round(best_results['mare'], 4):
        n_improve = n_improve + 1

    if n_improve > 2:
        best_results['step'] = step
        best_results['mae'] = mae
        best_results['rmse'] = rmse
        best_results['imae'] = imae
        best_results['irmse'] = irmse
        best_results['mare'] = mare

    log('Best results:', log_path)
    log('{:>8}  {:>8}  {:>8}  {:>8}  {:>8}  {:>8}'.format(
        'Step', 'MAE', 'RMSE', 'iMAE', 'iRMSE', 'MARE'),
        log_path)
    log('{:8}  {:8.3f}  {:8.3f}  {:8.3f}  {:8.3f}  {:8.3f}'.format(
        best_results['step'],
        best_results['mae'],
        best_results['rmse'],
        best_results['imae'],
        best_results['irmse'],
        best_results['mare']), log_path)

    return best_results

def run(image_path,
        sparse_depth_path,
        ground_truth_path,
        # Checkpoint settings
        restore_path,
        # Input settings
        normalized_image_range,
        outlier_removal_kernel_size,
        outlier_removal_threshold,
        # Spatial pyramid pool settings
        max_pool_sizes_spatial_pyramid_pool,
        n_convolution_spatial_pyramid_pool,
        n_filter_spatial_pyramid_pool,
        # ScaffNet settings
        encoder_type_scaffnet,
        n_filters_encoder_scaffnet,
        decoder_type_scaffnet,
        n_filters_decoder_scaffnet,
        min_predict_depth_scaffnet,
        max_predict_depth_scaffnet,
        # FusionNet network settings
        encoder_type_fusionnet,
        n_filters_encoder_image_fusionnet,
        n_filters_encoder_depth_fusionnet,
        decoder_type_fusionnet,
        n_filters_decoder_fusionnet,
        scale_match_method_fusionnet,
        scale_match_kernel_size_fusionnet,
        min_predict_depth_fusionnet,
        max_predict_depth_fusionnet,
        min_multiplier_depth_fusionnet,
        max_multiplier_depth_fusionnet,
        min_residual_depth_fusionnet,
        max_residual_depth_fusionnet,
        # Weight settings
        weight_initializer,
        activation_func,
        # Evaluation settings
        min_evaluate_depth,
        max_evaluate_depth,
        # Output settings
        output_path,
        save_outputs,
        keep_input_filenames,
        # Hardware settings
        device='cuda'):

    # Select device to run on
    if device == 'cuda' or device == 'gpu':
        device = torch.device('cuda')
    elif device == 'cpu':
        device = torch.device('cpu')
    else:
        raise ValueError('Unsupported device: {}'.format(device))

    '''
    Set up output paths
    '''
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    log_path = os.path.join(output_path, 'results.txt')
    output_dirpath = os.path.join(output_path, 'outputs')

    if save_outputs:
        # Create output directories
        image_dirpath = os.path.join(output_dirpath, 'image')
        input_depth_dirpath = os.path.join(output_dirpath, 'input_depth')
        output_depth_dirpath = os.path.join(output_dirpath, 'output_depth')
        sparse_depth_dirpath = os.path.join(output_dirpath, 'sparse_depth')
        ground_truth_dirpath = os.path.join(output_dirpath, 'ground_truth')

        dirpaths = [
            output_dirpath,
            image_dirpath,
            input_depth_dirpath,
            output_depth_dirpath,
            sparse_depth_dirpath,
            ground_truth_dirpath
        ]

        for dirpath in dirpaths:
            if not os.path.exists(dirpath):
                os.makedirs(dirpath)

    '''
    Load input paths and set up dataloaders
    '''
    image_paths = data_utils.read_paths(image_path)
    sparse_depth_paths = data_utils.read_paths(sparse_depth_path)

    n_sample = len(image_paths)

    is_available_ground_truth = False

    if ground_truth_path is not None:
        is_available_ground_truth = True
        ground_truth_paths = data_utils.read_paths(ground_truth_path)
    else:
        ground_truth_paths = [None] * n_sample

    input_paths = [
        sparse_depth_paths,
        ground_truth_paths
    ]

    for paths in input_paths:
        assert len(paths) == n_sample

    dataloader = torch.utils.data.DataLoader(
        datasets.FusionNetStandaloneInferenceDataset(
            image_paths=image_paths,
            sparse_depth_paths=sparse_depth_paths,
            ground_truth_paths=ground_truth_paths,
            load_image_triplets=False),
        batch_size=1,
        shuffle=False,
        num_workers=1,
        drop_last=False)

    transforms = Transforms(
        normalized_image_range=normalized_image_range)

    # Initialize outlier removal for sparse depth
    outlier_removal = OutlierRemoval(
        kernel_size=outlier_removal_kernel_size,
        threshold=outlier_removal_threshold)

    '''
    Set up the model
    '''
    # Build ScaffNet
    scaffnet_model = ScaffNetModel(
        max_pool_sizes_spatial_pyramid_pool=max_pool_sizes_spatial_pyramid_pool,
        n_convolution_spatial_pyramid_pool=n_convolution_spatial_pyramid_pool,
        n_filter_spatial_pyramid_pool=n_filter_spatial_pyramid_pool,
        encoder_type=encoder_type_scaffnet,
        n_filters_encoder=n_filters_encoder_scaffnet,
        decoder_type=decoder_type_scaffnet,
        n_filters_decoder=n_filters_decoder_scaffnet,
        n_output_resolution=1,
        weight_initializer=weight_initializer,
        activation_func=activation_func,
        min_predict_depth=min_predict_depth_scaffnet,
        max_predict_depth=max_predict_depth_scaffnet,
        device=device)

    # Build FusionNet
    fusionnet_model = FusionNetModel(
        encoder_type=encoder_type_fusionnet,
        n_filters_encoder_image=n_filters_encoder_image_fusionnet,
        n_filters_encoder_depth=n_filters_encoder_depth_fusionnet,
        decoder_type=decoder_type_fusionnet,
        n_filters_decoder=n_filters_decoder_fusionnet,
        scale_match_method=scale_match_method_fusionnet,
        scale_match_kernel_size=scale_match_kernel_size_fusionnet,
        min_predict_depth=min_predict_depth_fusionnet,
        max_predict_depth=max_predict_depth_fusionnet,
        min_multiplier_depth=min_multiplier_depth_fusionnet,
        max_multiplier_depth=max_multiplier_depth_fusionnet,
        min_residual_depth=min_residual_depth_fusionnet,
        max_residual_depth=max_residual_depth_fusionnet,
        weight_initializer=weight_initializer,
        activation_func=activation_func,
        device=device)

    # Restore model and set to evaluation mode
    fusionnet_model.restore_model(
        checkpoint_path=restore_path,
        scaffnet_model=scaffnet_model)

    scaffnet_model.eval()
    fusionnet_model.eval()

    parameters_scaffnet_model = scaffnet_model.parameters()
    parameters_fusionnet_model = fusionnet_model.parameters()

    '''
    Log input paths
    '''
    log('Input paths:', log_path)
    input_paths = [
        image_path,
        sparse_depth_path,
        ground_truth_path
    ]
    for path in input_paths:
        if path is not None:
            log(path, log_path)
    log('', log_path)

    log_input_settings(
        log_path,
        # Input settings
        normalized_image_range=normalized_image_range,
        outlier_removal_kernel_size=outlier_removal_kernel_size,
        outlier_removal_threshold=outlier_removal_threshold)

    log_scaffnet_settings(
        log_path,
        # Spatial pyramid pool settings
        max_pool_sizes_spatial_pyramid_pool=max_pool_sizes_spatial_pyramid_pool,
        n_convolution_spatial_pyramid_pool=n_convolution_spatial_pyramid_pool,
        n_filter_spatial_pyramid_pool=n_filter_spatial_pyramid_pool,
        # Depth network settings
        encoder_type=encoder_type_scaffnet,
        n_filters_encoder=n_filters_encoder_scaffnet,
        decoder_type=decoder_type_scaffnet,
        n_filters_decoder=n_filters_decoder_scaffnet,
        n_output_resolution=1,
        min_predict_depth=min_predict_depth_scaffnet,
        max_predict_depth=max_predict_depth_scaffnet,
        # Weight settings
        weight_initializer=weight_initializer,
        activation_func=activation_func,
        parameters_model=parameters_scaffnet_model)

    log_fusionnet_settings(
        log_path,
        # Depth network settings
        encoder_type=encoder_type_fusionnet,
        n_filters_encoder_image=n_filters_encoder_image_fusionnet,
        n_filters_encoder_depth=n_filters_encoder_depth_fusionnet,
        decoder_type=decoder_type_fusionnet,
        n_filters_decoder=n_filters_decoder_fusionnet,
        scale_match_method=scale_match_method_fusionnet,
        scale_match_kernel_size=scale_match_kernel_size_fusionnet,
        min_predict_depth=min_predict_depth_fusionnet,
        max_predict_depth=max_predict_depth_fusionnet,
        min_multiplier_depth=min_multiplier_depth_fusionnet,
        max_multiplier_depth=max_multiplier_depth_fusionnet,
        min_residual_depth=min_residual_depth_fusionnet,
        max_residual_depth=max_residual_depth_fusionnet,
        # Weight settings
        weight_initializer=weight_initializer,
        activation_func=activation_func,
        parameters_depth_model=parameters_fusionnet_model)

    log_evaluation_settings(
        log_path,
        min_evaluate_depth=min_evaluate_depth,
        max_evaluate_depth=max_evaluate_depth)

    log_system_settings(
        log_path,
        # Checkpoint settings
        checkpoint_path=output_path,
        depth_model_restore_path=restore_path,
        # Hardware settings
        device=device,
        n_thread=1)

    # Set up metrics in case groundtruth is available
    mae = np.zeros(n_sample)
    rmse = np.zeros(n_sample)
    imae = np.zeros(n_sample)
    irmse = np.zeros(n_sample)
    mare = np.zeros(n_sample)

    time_elapse = 0.0

    for idx, inputs in enumerate(dataloader):

        # Move inputs to device
        inputs = [
            in_.to(device) for in_ in inputs
        ]

        if is_available_ground_truth:
            image, sparse_depth, ground_truth = inputs
        else:
            image, sparse_depth = inputs

        time_start = time.time()

        with torch.no_grad():

            # Forward through ScaffNet
            input_depth = scaffnet_model.forward(sparse_depth)

            if 'uncertainty' in scaffnet_model.decoder_type:
                input_depth = input_depth[:, 0:1, :, :]

            # Validity map is where sparse depth is available
            validity_map = torch.where(
                sparse_depth > 0,
                torch.ones_like(sparse_depth),
                sparse_depth)

            # Remove outlier points and update sparse depth and validity map
            filtered_sparse_depth, _ = outlier_removal.remove_outliers(
                sparse_depth=sparse_depth,
                validity_map=validity_map)

            [image] = transforms.transform(
                images_arr=[image],
                random_transform_probability=0.0)

            # Forward through network
            output_depth = fusionnet_model.forward(
                image=image,
                input_depth=input_depth,
                sparse_depth=filtered_sparse_depth)

        time_elapse = time_elapse + (time.time() - time_start)

        # Convert to numpy
        output_depth = np.squeeze(output_depth.detach().cpu().numpy())

        # Save to output
        if save_outputs:
            image = np.transpose(np.squeeze(image.cpu().numpy()), (1, 2, 0))
            sparse_depth = np.squeeze(sparse_depth.cpu().numpy())
            input_depth = np.squeeze(input_depth.cpu().numpy())

            if keep_input_filenames:
                filename = os.path.basename(image_paths[idx])
            else:
                filename = '{:010d}.png'.format(idx)

            image_path = os.path.join(image_dirpath, filename)
            image = (255 * image).astype(np.uint8)
            Image.fromarray(image).save(image_path)

            input_depth_path = os.path.join(input_depth_dirpath, filename)
            data_utils.save_depth(input_depth, input_depth_path)

            output_depth_path = os.path.join(output_depth_dirpath, filename)
            data_utils.save_depth(output_depth, output_depth_path)

            sparse_depth_path = os.path.join(sparse_depth_dirpath, filename)
            data_utils.save_depth(sparse_depth, sparse_depth_path)

        if is_available_ground_truth:
            ground_truth = np.squeeze(ground_truth.cpu().numpy())

            if save_outputs:
                ground_truth_path = os.path.join(ground_truth_dirpath, filename)
                data_utils.save_depth(ground_truth, ground_truth_path)

            validity_mask = np.where(ground_truth > 0, 1, 0)
            min_max_mask = np.logical_and(
                ground_truth > min_evaluate_depth,
                ground_truth < max_evaluate_depth)
            mask = np.where(np.logical_and(validity_mask, min_max_mask) > 0)

            output_depth = output_depth[mask]
            ground_truth = ground_truth[mask]

            mae[idx] = eval_utils.mean_abs_err(1000.0 * output_depth, 1000.0 * ground_truth)
            rmse[idx] = eval_utils.root_mean_sq_err(1000.0 * output_depth, 1000.0 * ground_truth)
            imae[idx] = eval_utils.inv_mean_abs_err(0.001 * output_depth, 0.001 * ground_truth)
            irmse[idx] = eval_utils.inv_root_mean_sq_err(0.001 * output_depth, 0.001 * ground_truth)
            mare[idx] = eval_utils.mean_abs_rel_err(output_depth, ground_truth)

    # Compute total time elapse in ms
    time_elapse = time_elapse * 1000.0

    if is_available_ground_truth:
        mae_mean   = np.mean(mae)
        rmse_mean  = np.mean(rmse)
        imae_mean  = np.mean(imae)
        irmse_mean = np.mean(irmse)
        mare_mean = np.mean(mare)

        mae_std = np.std(mae)
        rmse_std = np.std(rmse)
        imae_std = np.std(imae)
        irmse_std = np.std(irmse)
        mare_std = np.std(mare)

        # Print evaluation results to console and file
        log('Evaluation results:', log_path)
        log('{:>8}  {:>8}  {:>8}  {:>8}  {:>8}'.format(
            'MAE', 'RMSE', 'iMAE', 'iRMSE', 'MARE'),
            log_path)
        log('{:8.3f}  {:8.3f}  {:8.3f}  {:8.3f}  {:8.3f}'.format(
            mae_mean, rmse_mean, imae_mean, irmse_mean, mare_mean),
            log_path)

        log('{:>8}  {:>8}  {:>8}  {:>8}  {:>8}'.format(
            '+/-', '+/-', '+/-', '+/-', '+/-'),
            log_path)
        log('{:8.3f}  {:8.3f}  {:8.3f}  {:8.3f}  {:8.3f}'.format(
            mae_std, rmse_std, imae_std, irmse_std, mare_std),
            log_path)

    # Log run time
    log('Total time: {:.2f} ms  Average time per sample: {:.2f} ms'.format(
        time_elapse, time_elapse / float(n_sample)))


def log_input_settings(log_path,
                       normalized_image_range,
                       outlier_removal_kernel_size,
                       outlier_removal_threshold,
                       n_batch=None,
                       n_height=None,
                       n_width=None,
                       cap_dataset_depth_method='none',
                       min_dataset_depth=-1,
                       max_dataset_depth=-1):

    batch_settings_text = ''
    batch_settings_vars = []

    if n_batch is not None:
        batch_settings_text = batch_settings_text + 'n_batch={}'
        batch_settings_vars.append(n_batch)

    batch_settings_text = \
        batch_settings_text + '  ' if len(batch_settings_text) > 0 else batch_settings_text

    if n_height is not None:
        batch_settings_text = batch_settings_text + 'n_height={}'
        batch_settings_vars.append(n_height)

    batch_settings_text = \
        batch_settings_text + '  ' if len(batch_settings_text) > 0 else batch_settings_text

    if n_width is not None:
        batch_settings_text = batch_settings_text + 'n_width={}'
        batch_settings_vars.append(n_width)

    if cap_dataset_depth_method is not None and cap_dataset_depth_method != 'none':
        log('cap_dataset_depth_method={}  min_dataset_depth={}  max_dataset_depth={}'.format(
            cap_dataset_depth_method, min_dataset_depth, max_dataset_depth),
            log_path)

    log('Input settings:', log_path)

    if len(batch_settings_vars) > 0:
        log(batch_settings_text.format(*batch_settings_vars),
            log_path)

    log('normalized_image_range={}'.format(normalized_image_range),
        log_path)
    log('outlier_removal_kernel_size={}  outlier_removal_threshold={:.2f}'.format(
        outlier_removal_kernel_size, outlier_removal_threshold),
        log_path)
    log('', log_path)

def log_fusionnet_settings(log_path,
                           # Depth network settings
                           encoder_type,
                           n_filters_encoder_image,
                           n_filters_encoder_depth,
                           decoder_type,
                           n_filters_decoder,
                           scale_match_method,
                           scale_match_kernel_size,
                           min_predict_depth,
                           max_predict_depth,
                           min_multiplier_depth,
                           max_multiplier_depth,
                           min_residual_depth,
                           max_residual_depth,
                           # Weight settings
                           weight_initializer,
                           activation_func,
                           parameters_depth_model=[],
                           parameters_pose_model=[]):

    # Computer number of parameters
    n_parameter_depth = sum(p.numel() for p in parameters_depth_model)
    n_parameter_pose = sum(p.numel() for p in parameters_pose_model)

    n_parameter = n_parameter_depth + n_parameter_pose

    n_parameter_text = 'n_parameter={}'.format(n_parameter)
    n_parameter_vars = []

    if n_parameter_depth > 0 :
        n_parameter_text = \
            n_parameter_text + '  ' if len(n_parameter_text) > 0 else n_parameter_text

        n_parameter_text = n_parameter_text + 'n_parameter_depth={}'
        n_parameter_vars.append(n_parameter_depth)

    if n_parameter_pose > 0 :
        n_parameter_text = \
            n_parameter_text + '  ' if len(n_parameter_text) > 0 else n_parameter_text

        n_parameter_text = n_parameter_text + 'n_parameter_pose={}'
        n_parameter_vars.append(n_parameter_pose)

    log('Depth network settings:', log_path)
    log('encoder_type={}'.format(encoder_type),
        log_path)
    log('n_filters_encoder_image={}'.format(n_filters_encoder_image),
        log_path)
    log('n_filters_encoder_depth={}'.format(n_filters_encoder_depth),
        log_path)
    log('decoder_type={}'.format(decoder_type),
        log_path)
    log('n_filters_decoder={}'.format(
        n_filters_decoder),
        log_path)
    log('scale_match_method={}  scale_match_kernel_size={}'.format(
        scale_match_method, scale_match_kernel_size),
        log_path)
    log('min_predict_depth={:.2f}  max_predict_depth={:.2f}'.format(
        min_predict_depth, max_predict_depth),
        log_path)
    log('min_multiplier_depth={:.2f}  max_multiplier_depth={:.2f}'.format(
        min_multiplier_depth, max_multiplier_depth),
        log_path)
    log('min_residual_depth={:.2f}  max_residual_depth={:.2f}'.format(
        min_residual_depth, max_residual_depth),
        log_path)
    log('', log_path)

    log('Weight settings:', log_path)
    log(n_parameter_text.format(*n_parameter_vars),
        log_path)
    log('weight_initializer={}  activation_func={}'.format(
        weight_initializer, activation_func),
        log_path)
    log('', log_path)

def log_training_settings(log_path,
                          # Training settings
                          n_batch,
                          n_train_sample,
                          n_train_step,
                          learning_rates,
                          learning_schedule,
                          # Augmentation settings
                          augmentation_probabilities,
                          augmentation_schedule,
                          augmentation_random_crop_type,
                          augmentation_random_crop_to_shape,
                          augmentation_random_flip_type,
                          augmentation_random_remove_points,
                          augmentation_random_noise_type,
                          augmentation_random_noise_spread,
                          augmentation_random_brightness,
                          augmentation_random_contrast,
                          augmentation_random_saturation,
                          # Network settings
                          freeze_scaffnet_modules=None):

    log('Training settings:', log_path)
    log('n_sample={}  n_epoch={}  n_step={}'.format(
        n_train_sample, learning_schedule[-1], n_train_step),
        log_path)
    log('learning_schedule=[%s]' %
        ', '.join('{}-{} : {}'.format(
            ls * (n_train_sample // n_batch), le * (n_train_sample // n_batch), v)
            for ls, le, v in zip([0] + learning_schedule[:-1], learning_schedule, learning_rates)),
        log_path)
    log('', log_path)

    if freeze_scaffnet_modules is not None:
        log('freeze_scaffnet_modules={}'.format(
            freeze_scaffnet_modules),
            log_path)
        log('', log_path)

    log('Augmentation settings:', log_path)
    log('augmentation_schedule=[%s]' %
        ', '.join('{}-{} : {}'.format(
            ls * (n_train_sample // n_batch), le * (n_train_sample // n_batch), v)
            for ls, le, v in zip([0] + augmentation_schedule[:-1], augmentation_schedule, augmentation_probabilities)),
        log_path)
    log('augmentation_random_crop_type={}'.format(augmentation_random_crop_type),
        log_path)
    log('augmentation_random_crop_to_shape={}'.format(augmentation_random_crop_to_shape),
        log_path)
    log('augmentation_random_flip_type={}'.format(augmentation_random_flip_type),
        log_path)
    log('augmentation_random_remove_points={}'.format(augmentation_random_remove_points),
        log_path)
    log('augmentation_random_noise_type={}  augmentation_random_noise_spread={}'.format(
        augmentation_random_noise_type, augmentation_random_noise_spread),
        log_path)
    log('augmentation_random_brightness={}'.format(augmentation_random_brightness),
        log_path)
    log('augmentation_random_contrast={}'.format(augmentation_random_contrast),
        log_path)
    log('augmentation_random_saturation={}'.format(augmentation_random_saturation),
        log_path)
    log('', log_path)

def log_loss_func_settings(log_path,
                           # Loss function settings
                           w_color=0.00,
                           w_structure=0.00,
                           w_sparse_depth=0.00,
                           w_smoothness=0.00,
                           w_prior_depth=0.00,
                           threshold_prior_depth=0.00,
                           w_weight_decay_depth=0.00,
                           w_weight_decay_pose=0.00):

    log('Loss function settings:', log_path)

    log('w_color={:.1e}  w_structure={:.1e}  w_sparse_depth={:.1e}'.format(
        w_color, w_structure, w_sparse_depth),
        log_path)
    log('w_smoothness={:.1e}'.format(w_smoothness),
        log_path)
    log('w_prior_depth={:.1e}  threshold_prior_depth={}'.format(
        w_prior_depth, threshold_prior_depth),
        log_path)

    log('w_weight_decay_depth={:.1e}  w_weight_decay_pose={:.1e}'.format(
        w_weight_decay_depth, w_weight_decay_pose),
        log_path)
    log('', log_path)

def log_evaluation_settings(log_path,
                            min_evaluate_depth,
                            max_evaluate_depth):

    log('Evaluation settings:', log_path)
    log('min_evaluate_depth={:.2f}  max_evaluate_depth={:.2f}'.format(
        min_evaluate_depth, max_evaluate_depth),
        log_path)
    log('', log_path)

def log_system_settings(log_path,
                        # Checkpoint settings
                        checkpoint_path,
                        n_step_per_checkpoint=None,
                        summary_event_path=None,
                        n_step_per_summary=None,
                        n_image_per_summary=None,
                        start_step_validation=None,
                        depth_model_restore_path=None,
                        pose_model_restore_path=None,
                        # Hardware settings
                        device=torch.device('cuda'),
                        n_thread=8):

    log('Checkpoint settings:', log_path)

    if checkpoint_path is not None:
        log('checkpoint_path={}'.format(checkpoint_path), log_path)

        if n_step_per_checkpoint is not None:
            log('n_step_per_checkpoint={}'.format(n_step_per_checkpoint), log_path)

        log('', log_path)

        summary_settings_text = ''
        summary_settings_vars = []

    if summary_event_path is not None:
        log('Tensorboard settings:', log_path)
        log('event_path={}'.format(summary_event_path), log_path)

    if n_step_per_summary is not None:
        summary_settings_text = summary_settings_text + 'n_step_per_summary={}'
        summary_settings_vars.append(n_step_per_summary)

        summary_settings_text = \
            summary_settings_text + '  ' if len(summary_settings_text) > 0 else summary_settings_text

    if n_image_per_summary is not None:
        summary_settings_text = summary_settings_text + 'n_image_per_summary={}'
        summary_settings_vars.append(n_image_per_summary)

        summary_settings_text = \
            summary_settings_text + '  ' if len(summary_settings_text) > 0 else summary_settings_text

    if len(summary_settings_text) > 0:
        log(summary_settings_text.format(*summary_settings_vars), log_path)

    if start_step_validation is not None:
        log('start_step_validation={}'.format(start_step_validation),
            log_path)

    if depth_model_restore_path is not None and depth_model_restore_path != '':
        log('depth_model_restore_path={}'.format(depth_model_restore_path),
            log_path)

    if pose_model_restore_path is not None and pose_model_restore_path != '':
        log('pose_model_restore_path={}'.format(pose_model_restore_path),
            log_path)

    log('', log_path)

    log('Hardware settings:', log_path)
    log('device={}'.format(device.type), log_path)
    log('n_thread={}'.format(n_thread), log_path)
    log('', log_path)
