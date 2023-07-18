#!/bin/bash

export CUDA_VISIBLE_DEVICES=1

python src/run_fusionnet_standalone.py \
--restore_path_scaffnet pretrained_models/scaffnet/scenenet/retrained/scaffnet.ckpt-scenenet \
--restore_path_fusionnet pretrained_models/fusionnet/void/retrained/fusionnet.ckpt-void \
--image_path testing/void/void_test_image_1500.txt \
--sparse_depth_path testing/void/void_test_sparse_depth_1500.txt \
--ground_truth_path testing/void/void_test_ground_truth_1500.txt \
--n_batch 8 \
--n_height 480 \
--n_width 640 \
--load_image_composite \
--network_type_scaffnet scaffnet32 \
--activation_func_scaffnet leaky_relu \
--n_filter_output_scaffnet 32 \
--min_predict_depth 0.1 \
--max_predict_depth 8.0 \
--pool_kernel_sizes_spp 5 7 9 11 13 \
--n_convolution_spp 3 \
--n_filter_spp 32 \
--network_type_fusionnet fusionnet05 \
--image_filter_pct 0.75 \
--depth_filter_pct 0.25 \
--activation_func_fusionnet leaky_relu \
--min_scale_depth 0.25 \
--max_scale_depth 4.00 \
--min_residual_depth -1000.0 \
--max_residual_depth 1000.0 \
--min_evaluate_depth 0.2 \
--max_evaluate_depth 5.0 \
--save_outputs \
--output_path pretrained_models/fusionnet/void/retrained/standalone/outputs/void1500 \
--n_thread 4
