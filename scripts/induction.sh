#!/bin/bash 

VOCAB_SIZE=10
MIN_LENGTH=9
MAX_LENGTH=9
SEED=6

echo "SEED=${SEED}";

python src/run.py \
     --dataset "induction" \
     --vocab_size "${VOCAB_SIZE}" \
     --dataset_size 20000 \
     --min_length "${MIN_LENGTH}" \
     --max_length "${MAX_LENGTH}" \
     --n_epochs 500 \
     --batch_size 512 \
     --lr "5e-2" \
     --gumbel_samples 1 \
     --sample_fn "gumbel_soft" \
     --tau_init 3.0 \
     --tau_end 0.01 \
     --tau_schedule "geomspace" \
     --n_vars_cat 1 \
     --n_vars_num 1 \
     --n_layers 2 \
     --n_heads_cat 1 \
     --n_heads_num 0 \
     --n_cat_mlps 0 \
     --n_num_mlps 0 \
     --attention_type "cat" \
     --rel_pos_bias "fixed" \
     --one_hot_embed \
     --count_only \
     --selector_width 0 \
     --seed "${SEED}" \
     --unique 1 \
     --unembed_mask 0 \
     --autoregressive \
     --save \
     --save_code \
     --output_dir "output/induction/s${SEED}";
