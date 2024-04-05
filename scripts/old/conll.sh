#!/bin/bash 

N_EPOCHS=50
N_LAYERS=2
N_VARS_CAT=4
N_HEADS_CAT=8
N_CAT_MLPS=1
SEED=0

python src/run.py \
     --dataset "conll_ner" \
     --vocab_size 10000 \
     --min_length 1 \
     --max_length 32 \
     --n_epochs "${N_EPOCHS}" \
     --batch_size 32 \
     --lr "1e-2" \
     --gumbel_samples 1 \
     --sample_fn "gumbel_soft" \
     --tau_init 3.0 \
     --tau_end 0.01 \
     --tau_schedule "geomspace" \
     --n_vars_cat "${N_VARS_CAT}" \
     --d_var 32 \
     --n_vars_num 1 \
     --n_layers "${N_LAYERS}" \
     --n_heads_cat "${N_HEADS_CAT}" \
     --n_heads_num 0 \
     --n_cat_mlps "${N_CAT_MLPS}" \
     --n_num_mlps 0 \
     --attention_type "cat" \
     --rel_pos_bias "fixed" \
     --dropout 0.5 \
     --mlp_vars_in 2 \
     --d_mlp 64 \
     --count_only \
     --selector_width 0 \
     --do_lower 0 \
     --replace_numbers 1 \
     --glove_embeddings "data/glove.840B.300d.txt" \
     --do_glove 1 \
     --pool_outputs 0 \
     --seed "${SEED}" \
     --save \
     --save_code \
     --output_dir "output/conll_ner/transformer_program/nvars${N_VARS_CAT}nheads${N_HEADS_CAT}nlayers${N_LAYERS}nmlps${N_CAT_MLPS}/epochs${N_EPOCHS}/s${SEED}";
