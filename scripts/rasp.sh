#!/bin/bash 

DATASET="hist"
VOCAB_SIZE=8
TRAIN_MIN_LENGTH=1
TEST_MIN_LENGTH=11
TRAIN_MAX_LENGTH=10
TEST_MAX_LENGTH=20
N_LAYERS=1
N_HEADS_CAT=2
N_HEADS_NUM=2
N_CAT_MLPS=1
N_NUM_MLPS=1
SEED=1


python ../src/run.py \
     --dataset "${DATASET}" \
     --vocab_size "${VOCAB_SIZE}" \
     --dataset_size 20000 \
     --train_min_length "${TRAIN_MIN_LENGTH}" \
     --train_max_length "${TRAIN_MAX_LENGTH}" \
     --test_min_length "${TEST_MIN_LENGTH}" \
     --test_max_length "${TEST_MAX_LENGTH}" \
     --n_epochs 250 \
     --batch_size 512 \
     --lr "5e-2" \
     --gumbel_samples 1 \
     --sample_fn "gumbel_soft" \
     --tau_init 3.0 \
     --tau_end 0.01 \
     --tau_schedule "geomspace" \
     --n_vars_cat 1 \
     --d_var "${TEST_MAX_LENGTH}" \
     --n_vars_num 1 \
     --n_layers "${N_LAYERS}" \
     --n_heads_cat "${N_HEADS_CAT}" \
     --n_heads_num "${N_HEADS_NUM}" \
     --n_cat_mlps "${N_CAT_MLPS}" \
     --n_num_mlps "${N_NUM_MLPS}" \
     --attention_type "cat" \
     --rel_pos_bias "fixed" \
     --one_hot_embed \
     --dropout 0.0 \
     --mlp_vars_in 2 \
     --d_mlp 64 \
     --count_only \
     --selector_width 0 \
     --seed "${SEED}" \
     --unique 1 \
     --save \
     --save_code \
     --device "cpu" \
     --output_dir "output/rasp/${DATASET}/vocab${VOCAB_SIZE}maxlen${MAX_LENGTH}/transformer_program/headsc${N_HEADS_CAT}headsn${N_HEADS_NUM}nlayers${N_LAYERS}cmlps${N_CAT_MLPS}nmlps${N_NUM_MLPS}/s${SEED}";

# --dataset "hist" --dataset_size 20000 --seed 1 --d_var 8 --n_heads_cat 2 --n_heads_num 2 --n_cat_mlps 1 --n_num_mlps 1 --n_layers 1 --one_hot_embed --count_only --save --save_code --device "cpu" --output_dir "output/rasp/hist/s1"