#!/bin/bash

export CUDA_VISIBLE_DEVICES=1

# $1 : path to directory containing checkpoints
# $2 : first checkpoint to evaluate
# $3 : increment between checkpoints
# $4 : last checkpoint to evaluate

for n in $(seq $2 $3 $4)
do
python src/run_fusionnet.py \
--restore_path $1/model.ckpt-$n \
--image_path testing/void/void_test_image_1500.txt \
--input_depth_path testing/void/void_test_predict_depth_1500.txt \
--sparse_depth_path testing/void/void_test_sparse_depth_1500.txt \
--ground_truth_path testing/void/void_test_ground_truth_1500.txt \
--n_batch 8 \
--n_height 352 \
--n_width 1216 \
--network_type fusionnet05 \
--image_filter_pct 0.75 \
--depth_filter_pct 0.25 \
--activation_func leaky_relu \
--min_predict_depth 1.5 \
--max_predict_depth 100.0 \
--min_scale_depth 0.25 \
--max_scale_depth 4.00 \
--min_residual_depth -1000.0 \
--max_residual_depth 1000.0 \
--min_evaluate_depth 0.0 \
--max_evaluate_depth 100.0 \
--output_path $1/outputs/void1500\
--n_thread 4
done