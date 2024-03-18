#!/bin/bash 

N_EPOCHS=10
N_LAYERS=2
N_VARS_CAT=4
N_HEADS_CAT=4
N_CAT_MLPS=1
SEED=0

python src/run.py \
     --dataset "${DATASET}" \
     --vocab_size 10000 \
     --min_length 1 \
     --max_length 64 \
     --n_epochs "${N_EPOCHS}" \
     --batch_size 128 \
     --lr "5e-2" \
     --gumbel_samples 1 \
     --sample_fn "gumbel_soft" \
     --tau_init 3.0 \
     --tau_end 0.01 \
     --tau_schedule "geomspace" \
     --n_vars_cat "${N_VARS_CAT}" \
     --d_var 64 \
     --n_vars_num 1 \
     --n_layers "${N_LAYERS}" \
     --n_heads_cat "${N_HEADS_CAT}" \
     --n_heads_num 0 \
     --n_cat_mlps "${N_CAT_MLPS}" \
     --n_num_mlps 0 \
     --attention_type "cat" \
     --rel_pos_bias "fixed" \
     --dropout 0.0 \
     --mlp_vars_in 2 \
     --d_mlp 64 \
     --count_only \
     --selector_width 0 \
     --do_lower 0 \
     --replace_numbers 0 \
     --glove_embeddings "data/glove.840B.300d.txt" \
     --do_glove 1 \
     --pool_outputs 1 \
     --seed "${SEED}" \
     --save \
     --save_code \
     --output_dir "output/classification/${DATASET}/transformer_program/nvars${N_VARS_CAT}nheads${N_HEADS_CAT}nlayers${N_LAYERS}nmlps${N_CAT_MLPS}/epochs${N_EPOCHS}/s${SEED}";
