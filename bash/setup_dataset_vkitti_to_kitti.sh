#!bin/bash

export CUDA_VISIBLE_DEVICES=1

python setup/setup_dataset_syn2real.py \
--restore_path pretrained_models/scaffnet/vkitti/retrained/scaffnet.ckpt-vkitti \
--train_sparse_depth_path training/kitti/kitti_train_sparse_depth-clean.txt \
--val_sparse_depth_path validation/kitti/kitti_val_sparse_depth.txt \
--test_sparse_depth_path testing/kitti/kitti_test_sparse_depth.txt \
--network_type scaffnet32 \
--activation_func leaky_relu \
--n_filter_output 32 \
--pool_kernel_sizes_spp 5 7 9 11 \
--n_convolution_spp 3 \
--min_predict_depth 1.5 \
--max_predict_depth 100.0 \
--train_output_depth_path training/kitti/kitti_train_predict_depth-clean.txt \
--val_output_depth_path validation/kitti/kitti_val_predict_depth.txt \
--test_output_depth_path testing/kitti/kitti_test_predict_depth.txt \
--n_thread 2
