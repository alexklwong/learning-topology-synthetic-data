#!/bin/bash

export CUDA_VISIBLE_DEVICES=1

# $1 : path to directory containing checkpoints
# $2 : first checkpoint to evaluate
# $3 : increment between checkpoints
# $4 : last checkpoint to evaluate

for n in $(seq $2 $3 $4)
do
python src/run_scaffnet.py \
--restore_path $1/model.ckpt-$n \
--sparse_depth_path \
testing/void/void_test_sparse_depth_1500.txt \
--ground_truth_path \
testing/void/void_test_ground_truth_1500.txt \
--n_batch 8 \
--n_height 480 \
--n_width 640 \
--depth_load_multiplier 256.0 \
--network_type scaffnet32 \
--activation_func leaky_relu \
--n_filter_output 32 \
--pool_kernel_sizes_spp 5 7 9 11 \
--n_convolution_spp 3 \
--n_filter_spp 32 \
--min_predict_depth 0.1 \
--max_predict_depth 8.0 \
--min_evaluate_depth 0.0 \
--max_evaluate_depth 100.0 \
--output_path $1/outputs \
--n_thread 4
done
