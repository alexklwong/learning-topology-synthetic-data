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
--image_path testing/void/void_test_image_150.txt \
--input_depth_path testing/void/void_test_predict_depth_150.txt \
--sparse_depth_path testing/void/void_test_sparse_depth_150.txt \
--ground_truth_path testing/void/void_test_ground_truth_150.txt \
--n_batch 8 \
--n_height 480 \
--n_width 640 \
--load_image_composite \
--network_type fusionnet05 \
--image_filter_pct 0.75 \
--depth_filter_pct 0.25 \
--activation_func leaky_relu \
--min_predict_depth 0.1 \
--max_predict_depth 8.0 \
--min_scale_depth 0.25 \
--max_scale_depth 4.00 \
--min_residual_depth -1000.0 \
--max_residual_depth 1000.0 \
--min_evaluate_depth 0.2 \
--max_evaluate_depth 5.0 \
--output_path $1/outputs/void150 \
--n_thread 4
done
