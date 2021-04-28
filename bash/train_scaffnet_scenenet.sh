#!bin/bash

export CUDA_VISIBLE_DEVICES=0

python src/train_scaffnet.py \
--train_sparse_depth_path training/scenenet/scenenet_train_sparse_depth_corner-1.txt \
--train_ground_truth_path training/scenenet/scenenet_train_ground_truth_corner-1.txt \
--val_sparse_depth_path testing/void/void_test_sparse_depth_1500.txt \
--val_ground_truth_path testing/void/void_test_ground_truth_1500.txt \
--n_height 192 \
--n_width 288 \
--depth_load_multiplier 256 \
--min_dataset_depth 0.0 \
--max_dataset_depth 8.0 \
--augmentation_random_horizontal_crop \
--augmentation_random_vertical_crop \
--n_epoch 5 \
--learning_rates 5.00e-5 2.50e-5 1.00e-5 \
--learning_schedule 2 4 \
--loss_func l1_norm \
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
--n_checkpoint 5000 \
--checkpoint_path trained_scaffnet/scenenet/scaffnet32
