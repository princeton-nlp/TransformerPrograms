#!/bin/bash 

D_MODEL=64
N_LAYERS=2
N_HEADS=4
SEED=0

python src/run.py \
     --dataset "${DATASET}" \
     --standard \
     --vocab_size 10000 \
     --min_length 1 \
     --max_length 64 \
     --n_epochs 100 \
     --batch_size 128 \
     --lr "5e-3" \
     --d_model "${D_MODEL}" \
     --n_heads "${N_HEADS}" \
     --n_layers "${N_LAYERS}" \
     --d_mlp 64 \
     --dropout 0.5 \
     --max_grad_norm 5.0 \
     --do_lower 0 \
     --replace_numbers 0 \
     --glove_embeddings "data/glove.840B.300d.txt" \
     --do_glove "${DO_GLOVE}" \
     --pool_outputs 1 \
     --seed "${SEED}" \
     --output_dir "output/classification/${DATASET}/standard_transformer/dmodel${D_MODEL}nheads${N_CAT}nlayers${N_LAYERS}/s${SEED}";
