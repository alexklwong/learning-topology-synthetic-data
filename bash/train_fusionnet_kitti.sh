#!bin/bash

export CUDA_VISIBLE_DEVICES=0

python src/train_fusionnet.py \
--train_image_path training/kitti/kitti_train_image-clean.txt \
--train_input_depth_path training/kitti/kitti_train_predict_depth-clean.txt \
--train_sparse_depth_path training/kitti/kitti_train_sparse_depth-clean.txt \
--train_intrinsics_path training/kitti/kitti_train_intrinsics-clean.txt \
--val_image_path validation/kitti/kitti_val_image.txt \
--val_input_depth_path validation/kitti/kitti_val_predict_depth.txt \
--val_sparse_depth_path validation/kitti/kitti_val_sparse_depth.txt \
--val_ground_truth_path validation/kitti/kitti_val_ground_truth.txt \
--n_batch 8 \
--n_height 320 \
--n_width 768 \
--crop_type bottom \
--augmentation_random_horizontal_crop \
--network_type fusionnet05 \
--image_filter_pct 0.75 \
--depth_filter_pct 0.25 \
--activation_func leaky_relu \
--min_predict_depth 1.5 \
--max_predict_depth 1.00 \
--min_scale_depth 0.25 \
--max_scale_depth 4.00 \
--min_residual_depth -1000.0 \
--max_residual_depth 1000.0 \
--learning_rates 2.00e-4 1.00e-4 0.50e-4 \
--learning_schedule 18 24 \
--n_epoch 30 \
--validity_map_color nonsparse \
--w_color 0.20 \
--w_structure 0.80 \
--w_sparse_depth 0.10 \
--w_smoothness 0.01 \
--w_prior_depth 0.10 \
--residual_threshold_prior_depth 0.40 \
--rotation_param euler \
--min_evaluate_depth 0.0 \
--max_evaluate_depth 100.0 \
--checkpoint_path trained_fusionnet/kitti/fusionnet05 \
--n_checkpoint 5000 \
--n_summary 5000 \
--n_thread 8