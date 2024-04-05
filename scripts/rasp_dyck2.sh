#!/bin/bash 

for RUN in 0 1 2 3 4
do
     python ../src/run.py \
          --dataset "dyck2" \
          --vocab_size 2 \
          --dataset_size 20000 \
          --train_min_length 16 \
          --train_max_length 16 \
          --test_min_length 16 \
          --test_max_length 16 \
          --d_var 16 \
          --n_heads_cat 2 \
          --n_heads_num 2 \
          --n_cat_mlps 2 \
          --n_num_mlps 2 \
          --n_layers 3 \
          --one_hot_embed \
          --count_only \
          --autoregressive \
          --save \
          --save_code \
          --device "cpu" \
          --output_dir "output/rasp/dyck2/r${RUN}";
done