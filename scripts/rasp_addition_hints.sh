#!/bin/bash 

for RUN in 0 1 2 3 4
do
     python ../src/run.py \
          --dataset "addition_hints" \
          --vocab_size 10 \
          --dataset_size 20000 \
          --train_min_length 1 \
          --train_max_length 4 \
          --test_min_length 1 \
          --test_max_length 4 \
          --d_var 16 \
          --n_heads_cat 4 \
          --n_heads_num 4 \
          --n_cat_mlps 2 \
          --n_num_mlps 2 \
          --n_layers 3 \
          --one_hot_embed \
          --count_only \
          --save \
          --device "cpu" \
          --output_dir "output/rasp/addition_hints/r${RUN}";
done