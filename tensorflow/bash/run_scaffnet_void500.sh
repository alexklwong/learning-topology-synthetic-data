#!/bin/bash

export CUDA_VISIBLE_DEVICES=1

python src/run_scaffnet.py \
--restore_path pretrained_models/scaffnet/scenenet/retrained/scaffnet.ckpt-scenenet \
--sparse_depth_path testing/void/void_test_sparse_depth_500.txt \
--ground_truth_path testing/void/void_test_ground_truth_500.txt \
--n_batch 8 \
--n_height 480 \
--n_width 640 \
--network_type scaffnet32 \
--activation_func leaky_relu \
--n_filter_output 32 \
--pool_kernel_sizes_spp 5 7 9 11 13 \
--n_convolution_spp 3 \
--n_filter_spp 32 \
--min_predict_depth 0.1 \
--max_predict_depth 8.0 \
--min_evaluate_depth 0.2 \
--max_evaluate_depth 5.0 \
--save_outputs \
--output_path pretrained_models/scaffnet/scenenet/retrained/outputs/void500 \
--n_thread 4
