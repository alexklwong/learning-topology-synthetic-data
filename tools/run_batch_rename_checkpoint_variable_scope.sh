#!/bin/bash

export CUDA_VISIBLE_DEVICES=1

# $1 : path to directory containing checkpoints
# $2 : first checkpoint to evaluate
# $3 : increment between checkpoints
# $4 : last checkpoint to evaluate
# $5 : variable scope
# $6 : new variable scope

for n in $(seq $2 $3 $4)
do
python tools/rename_checkpoint_variable_scope.py \
--checkpoint_load_path $1/model.ckpt-$n \
--checkpoint_save_path $1/model.ckpt-$n \
--variable_scope $5 \
--new_variable_scope $6
done
