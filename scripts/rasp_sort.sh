#!/bin/bash 

for RUN in 0 1 2 3 4
do
     python ../src/run.py \
          --dataset "sort" \
          --dataset_size 20000 \
          --train_min_length 1 \
          --train_max_length 8 \
          --test_min_length 1 \
          --test_max_length 8 \
          --d_var 8 \
          --n_heads_cat 4 \
          --n_heads_num 4 \
          --n_cat_mlps 2 \
          --n_num_mlps 2 \
          --n_layers 3 \
          --one_hot_embed \
          --count_only \
          --save \
          --save_code \
          --device "cpu" \
          --output_dir "output/rasp/sort/r${RUN}";
done