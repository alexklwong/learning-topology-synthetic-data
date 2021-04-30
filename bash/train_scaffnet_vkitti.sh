#!bin/bash

export CUDA_VISIBLE_DEVICES=0

python src/train_scaffnet.py \
--train_sparse_depth_path training/vkitti/vkitti_train_sparse_depth.txt \
--train_ground_truth_path training/vkitti/vkitti_train_ground_truth.txt \
--val_sparse_depth_path validation/kitti/kitti_val_sparse_depth.txt \
--val_ground_truth_path validation/kitti/kitti_val_ground_truth.txt \
--n_batch 8 \
--n_height 320 \
--n_width 768 \
--depth_load_multiplier 256 \
--min_dataset_depth 1.5 \
--max_dataset_depth 80.0 \
--augmentation_random_horizontal_crop \
--augmentation_random_vertical_crop \
--n_epoch 50 \
--learning_rates 4.00e-4 3.00e-4 2.00e-4 1.00e-4 \
--learning_schedule 8 20 30 \
--loss_func l1_norm \
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
--n_checkpoint 5000 \
--checkpoint_path trained_scaffnet/vkitti/scaffnet32
