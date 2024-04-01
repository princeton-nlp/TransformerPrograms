#!/bin/bash 

for SEED in 0 1 2 3 4
do
     python ../src/run.py \
          --dataset "hist" \
          --dataset_size 20000 \
          --seed "${SEED}" \
          --d_var 8 \
          --n_heads_cat 2 \
          --n_heads_num 2 \
          --n_cat_mlps 1 \
          --n_num_mlps 1 \
          --n_layers 1 \
          --one_hot_embed \
          --count_only \
          --save \
          --save_code \
          --device "cpu" \
          --output_dir "output/rasp/hist/s${SEED}";
done          

# --dataset "hist" --dataset_size 20000 --seed 1 --d_var 8 --n_heads_cat 2 --n_heads_num 2 --n_cat_mlps 1 --n_num_mlps 1 --n_layers 1 --one_hot_embed --count_only --save --save_code --device "cpu" --output_dir "output/rasp/hist/s1"