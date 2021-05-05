#!bin/bash

export CUDA_VISIBLE_DEVICES=0

python src/train_fusionnet.py \
--train_image_path training/void/void_train_image_1500.txt \
--train_input_depth_path training/void/void_train_predict_depth_1500.txt \
--train_sparse_depth_path training/void/void_train_sparse_depth_1500.txt \
--train_intrinsics_path training/void/void_train_intrinsics_1500.txt \
--val_image_path testing/void/void_val_image_1500.txt \
--val_input_depth_path testing/void/void_val_predict_depth_1500.txt \
--val_sparse_depth_path testing/void/void_val_sparse_depth_1500.txt \
--val_ground_truth_path testing/void/void_val_ground_truth_1500.txt \
--n_batch 8 \
--n_height 480 \
--n_width 640 \
--crop_type center \
--augmentation_random_horizontal_crop \
--augmentation_random_vertical_crop \
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
--learning_rates 5.00e-5 2.50e-5 1.00e-4 \
--learning_schedule 6 8 \
--n_epoch 10 \
--validity_map_color nonsparse \
--w_color 0.20 \
--w_structure 0.80 \
--w_sparse_depth 1.00 \
--w_smoothness 0.40 \
--w_prior_depth 0.10 \
--residual_threshold_prior_depth 0.30 \
--rotation_param euler \
--min_evaluate_depth 0.2 \
--max_evaluate_depth 5.0 \
--checkpoint_path trained_fusionnet/void/fusionnet05 \
--n_checkpoint 5000 \
--n_summary 5000 \
--n_thread 8