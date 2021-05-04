#!/bin/bash

export CUDA_VISIBLE_DEVICES=1

python src/run_scaffnet.py \
--restore_path pretrained_models/scaffnet/vkitti/retrained/scaffnet.ckpt-vkitti \
--sparse_depth_path validation/kitti/kitti_val_sparse_depth.txt \
--ground_truth_path validation/kitti/kitti_val_ground_truth.txt \
--n_batch 8 \
--n_height 352 \
--n_width 1216 \
--network_type scaffnet32 \
--activation_func leaky_relu \
--n_filter_output 32 \
--pool_kernel_sizes_spp 5 7 9 11 \
--n_convolution_spp 3 \
--n_filter_spp 32 \
--min_predict_depth 1.5 \
--max_predict_depth 100.0 \
--min_evaluate_depth 0.0 \
--max_evaluate_depth 100.0 \
--save_outputs \
--output_path pretrained_models/scaffnet/vkitti/retrained/outputs/kitti \
--n_thread 4
