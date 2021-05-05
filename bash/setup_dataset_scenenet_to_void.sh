#!bin/bash

export CUDA_VISIBLE_DEVICES=1

python setup/setup_dataset_syn2real.py \
--restore_path pretrained_models/scaffnet/scenenet/retrained/scaffnet.ckpt-scenenet \
--train_sparse_depth_path training/void/void_train_sparse_depth_1500.txt \
--val_sparse_depth_path testing/void/void_test_sparse_depth_1500.txt \
--test_sparse_depth_path testing/void/void_test_sparse_depth_1500.txt \
--network_type scaffnet32 \
--activation_func leaky_relu \
--n_filter_output 32 \
--pool_kernel_sizes_spp 5 7 9 11 13 \
--n_convolution_spp 3 \
--min_predict_depth 0.1 \
--max_predict_depth 8.0 \
--train_output_depth_path training/void/void_train_predict_depth_1500.txt \
--val_output_depth_path testing/void/void_test_predict_depth_1500.txt \
--test_output_depth_path testing/void/void_test_predict_depth_1500.txt \
--n_thread 2

python setup/setup_dataset_syn2real.py \
--restore_path pretrained_models/scaffnet/scenenet/retrained/scaffnet.ckpt-scenenet \
--train_sparse_depth_path training/void/void_train_sparse_depth_500.txt \
--val_sparse_depth_path testing/void/void_test_sparse_depth_500.txt \
--test_sparse_depth_path testing/void/void_test_sparse_depth_500.txt \
--network_type scaffnet32 \
--activation_func leaky_relu \
--n_filter_output 32 \
--pool_kernel_sizes_spp 5 7 9 11 13 \
--n_convolution_spp 3 \
--min_predict_depth 0.1 \
--max_predict_depth 8.0 \
--train_output_depth_path training/void/void_train_predict_depth_500.txt \
--val_output_depth_path testing/void/void_test_predict_depth_500.txt \
--test_output_depth_path testing/void/void_test_predict_depth_500.txt \
--n_thread 2

python setup/setup_dataset_syn2real.py \
--restore_path pretrained_models/scaffnet/scenenet/retrained/scaffnet.ckpt-scenenet \
--train_sparse_depth_path training/void/void_train_sparse_depth_150.txt \
--val_sparse_depth_path testing/void/void_test_sparse_depth_150.txt \
--test_sparse_depth_path testing/void/void_test_sparse_depth_150.txt \
--network_type scaffnet32 \
--activation_func leaky_relu \
--n_filter_output 32 \
--pool_kernel_sizes_spp 5 7 9 11 13 \
--n_convolution_spp 3 \
--min_predict_depth 0.1 \
--max_predict_depth 8.0 \
--train_output_depth_path training/void/void_train_predict_depth_150.txt \
--val_output_depth_path testing/void/void_test_predict_depth_150.txt \
--test_output_depth_path testing/void/void_test_predict_depth_150.txt \
--n_thread 2
