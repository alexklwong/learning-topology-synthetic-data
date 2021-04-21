export CUDA_VISIBLE_DEVICES=1

python setup/setup_dataset_syn2real.py \
--restore_path \
best_models_vkitti/linear_pool5-7-9-11-p3_k5_out32_l1norm_lr0-4e4_10-3e4_20-2e4_29-1e4_50_gt15-80_randcrop_4/model.ckpt-193000 \
--train_sparse_depth_path \
training/kitti/kitti_train_sparse_depth-clean.txt \
--train_validity_map_path \
training/kitti/kitti_train_validity_map-clean.txt \
--val_sparse_depth_path \
validation/kitti/kitti_val_sparse_depth.txt \
--val_validity_map_path \
validation/kitti/kitti_val_validity_map.txt \
--test_sparse_depth_path \
testing/kitti/kitti_test_validity_map.txt \
--test_validity_map_path \
testing/kitti/kitti_test_validity_map.txt \
--input_root_dirpath \
data/kitti_depth_completion \
--output_root_dirpath \
data/kitti_depth_completion_learning_topology \
--depth_load_multiplier 256.0 \
--min_dataset_depth 0 \
--max_dataset_depth 100.0 \
--network_type scaffnet32 \
--activation_func leaky_relu \
--output_func linear \
--n_filter_output 32 \
--pool_kernel_sizes_spp 5 7 9 11 \
--n_convolution_spp 3 \
--min_predict_depth 1.5 \
--max_predict_depth 100.0 \
--train_output_depth_path \
training/kitti/kitti_train_predict_depth-clean.txt \
--val_output_depth_path \
validation/kitti/kitti_val_predict_depth-clean.txt \
--test_output_depth_path \
testing/kitti/kitti_test_predict_depth-clean.txt \
--n_thread 2
