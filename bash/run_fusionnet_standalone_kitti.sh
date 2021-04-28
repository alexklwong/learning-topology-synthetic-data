#!/bin/bash

export CUDA_VISIBLE_DEVICES=1

python src/run_fusionnet_standalone.py \
--restore_path_scaffnet \
pretrained_models/scaffnet/vkitti/scaffnet.ckpt-vkitti \
--restore_path_fusionnetnet \
pretrained_models/fusionnet/kitti/fusionnet.ckpt-kitti \
--image_path \
validation/kitti/kitti_val_image.txt \
--sparse_depth_path \
validation/kitti/kitti_val_sparse_depth.txt \
--ground_truth_path \
validation/kitti/kitti_val_ground_truth.txt \
--n_batch 8 \
--n_height 352 \
--n_width 1216 \
--depth_load_multiplier 256.0 \
--load_image_composite \
--network_type_scaffnet scaffnet32 \
--activation_func_scaffnet leaky_relu \
--n_filter_output_scaffnet 32 \
--pool_kernel_sizes_spp 5 7 9 11 \
--n_convolution_spp 3 \
--n_filter_spp 32 \
--network_type_fusionnet fusionnet05 \
--image_filter_pct 0.75 \
--depth_filter_pct 0.25 \
--activation_func_fusionnet leaky_relu \
--min_predict_depth 1.5 \
--max_predict_depth 1.00 \
--min_scale_depth 0.25 \
--max_scale_depth 4.00 \
--min_residual_depth -1000.0 \
--max_residual_depth 1000.0 \
--min_evaluate_depth 0.0 \
--max_evaluate_depth 100.0 \
--save_outputs \
--output_path \
pretrained_models/fusionnet/kitti/standalone/outputs \
--n_thread 4
