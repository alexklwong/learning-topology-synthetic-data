#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

python src/run_fusionnet.py \
--image_path validation/kitti/kitti_val_image.txt \
--sparse_depth_path validation/kitti/kitti_val_sparse_depth.txt \
--ground_truth_path validation/kitti/kitti_val_ground_truth.txt \
--restore_path pretrained_models/fusionnet/void/fusionnet-kitti.pth \
--normalized_image_range 0 1 \
--outlier_removal_kernel_size 7 \
--outlier_removal_threshold 1.5 \
--max_pool_sizes_spatial_pyramid_pool 13 17 19 21 25 \
--n_convolution_spatial_pyramid_pool 3 \
--n_filter_spatial_pyramid_pool 8 \
--encoder_type_scaffnet vggnet08 spatial_pyramid_pool batch_norm \
--n_filters_encoder_scaffnet 16 32 64 128 256 \
--decoder_type_scaffnet multi-scale batch_norm \
--n_filters_decoder_scaffnet 256 128 128 64 32 \
--min_predict_depth_scaffnet 1.5 \
--max_predict_depth_scaffnet 80.0 \
--encoder_type_fusionnet vggnet08 \
--n_filters_encoder_image_fusionnet 48 96 192 384 384 \
--n_filters_encoder_depth_fusionnet 16 32 64 128 128 \
--decoder_type_fusionnet multi-scale \
--n_filters_decoder_fusionnet 256 128 128 64 32 \
--scale_match_method_fusionnet local_scale \
--scale_match_kernel_size_fusionnet 5 \
--min_predict_depth_fusionnet 1.5 \
--max_predict_depth_fusionnet 100.0 \
--min_multiplier_depth_fusionnet 0.25 \
--max_multiplier_depth_fusionnet 4.00 \
--min_residual_depth_fusionnet -1000.0 \
--max_residual_depth_fusionnet 1000.0 \
--weight_initializer xavier_normal \
--activation_func leaky_relu \
--min_evaluate_depth 1e-3 \
--max_evaluate_depth 100.0 \
--output_path pretrained_models/fusionnet/kitti/evaluation_results/kitti-val \
--save_outputs \
--keep_input_filenames \
--device cuda
