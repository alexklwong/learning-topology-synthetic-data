#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

python src/run_scaffnet.py \
--sparse_depth_path validation/kitti/kitti_val_sparse_depth.txt \
--ground_truth_path validation/kitti/kitti_val_ground_truth.txt \
--restore_path \
pretrained_models/scaffnet/vkitti/scaffnet-vkitti.pth \
--max_pool_sizes_spatial_pyramid_pool 13 17 19 21 25 \
--n_convolution_spatial_pyramid_pool 3 \
--n_filter_spatial_pyramid_pool 8 \
--max_pool_sizes_spatial_pyramid_pool 13 17 19 21 25 \
--n_convolution_spatial_pyramid_pool 3 \
--n_filter_spatial_pyramid_pool 8 \
--encoder_type vggnet08 spatial_pyramid_pool batch_norm \
--n_filters_encoder 16 32 64 128 256 \
--decoder_type multi-scale batch_norm \
--n_filters_decoder 256 128 128 64 32 \
--min_predict_depth 1.5 \
--max_predict_depth 80.0 \
--weight_initializer xavier_normal \
--activation_func leaky_relu \
--min_evaluate_depth 1e-3 \
--max_evaluate_depth 100.0 \
--output_path \
pretrained_models/scaffnet/vkitti/evaluation_results/kitti-val \
--save_outputs \
--keep_input_filenames \
--device cuda
