#!/bin/bash

export CUDA_VISIBLE_DEVICES=1

python src/run_fusionnet.py \
--restore_path \
pretrained_models/fusionnet/kitti/paper/fusionnet.ckpt-kitti \
--image_path \
validation/kitti/kitti_val_image.txt \
--input_depth_path \
validation/kitti/kitti_val_predict_depth.txt \
--sparse_depth_path \
validation/kitti/kitti_val_sparse_depth.txt \
--ground_truth_path \
validation/kitti/kitti_val_ground_truth.txt \
--n_batch 8 \
--n_height 352 \
--n_width 1216 \
--load_image_composite \
--network_type fusionnet05 \
--image_filter_pct 0.75 \
--depth_filter_pct 0.25 \
--activation_func leaky_relu \
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
pretrained_models/fusionnet/kitti/paper/outputs \
--n_thread 4
