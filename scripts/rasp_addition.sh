#!/bin/bash 

# for SEED in 0 1 2 3 4
# do
python ../src/run.py \
     --dataset "add" \
     --dataset_size 20000 \
     --max_length 4 \
     --d_var 4 \
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
     --output_dir "output/rasp/add/s0";
# done          

# --dataset "add" --dataset_size 20000 --max_length 4 --d_var 4 --n_heads_cat 4 --n_heads_num 4 --n_cat_mlps 2 --n_num_mlps 2 --n_layers 3 --one_hot_embed --count_only --save --save_code --device "cpu" --output_dir "output/rasp/add/s0"