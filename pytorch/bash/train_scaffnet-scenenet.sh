#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

python src/train_scaffnet.py \
--train_sparse_depth_path training/scenenet/scenenet_train_sparse_depth_corner.txt \
--train_ground_truth_path training/scenenet/scenenet_train_ground_truth_corner.txt \
--val_sparse_depth_path testing/void/void_test_sparse_depth_1500.txt \
--val_ground_truth_path testing/void/void_test_ground_truth_1500.txt \
--n_batch 12 \
--n_height 192 \
--n_width 256 \
--cap_dataset_depth_method set_to_max \
--min_dataset_depth 0.0 \
--max_dataset_depth 10.0 \
--max_pool_sizes_spatial_pyramid_pool 13 17 19 21 25 \
--n_convolution_spatial_pyramid_pool 3 \
--n_filter_spatial_pyramid_pool 8 \
--encoder_type vggnet08 spatial_pyramid_pool batch_norm \
--n_filters_encoder 16 32 64 128 256 \
--decoder_type multi-scale batch_norm \
--n_filters_decoder 256 128 128 64 32 \
--min_predict_depth 0.1 \
--max_predict_depth 10.0 \
--weight_initializer xavier_normal \
--activation_func leaky_relu \
--freeze_network_modules none \
--learning_rates 1e-4 5e-5 \
--learning_schedule 4 10 \
--augmentation_probabilities 1.00 \
--augmentation_schedule -1 \
--augmentation_random_crop_type horizontal bottom anchored \
--augmentation_random_flip_type horizontal \
--loss_func supervised_l1_normalized \
--w_supervised 1.00 \
--w_weight_decay 0.00 \
--min_evaluate_depth 0.2 \
--max_evaluate_depth 5.0 \
--n_step_per_summary 5000 \
--n_image_per_summary 4 \
--n_step_per_checkpoint 5000 \
--checkpoint_path trained_scaffnet/scenenet/scaffnet_vgg08 \
--start_step_validation 15000 \
--device cuda \
--n_thread 8
