#!/bin/bash 

D_MODEL=128
N_LAYERS=3
N_HEADS=4
SEED=0

python src/run.py \
     --dataset "conll_ner" \
     --standard \
     --vocab_size 10000 \
     --min_length 1 \
     --max_length 32 \
     --n_epochs 100 \
     --batch_size 32 \
     --lr "5e-3" \
     --d_model "${D_MODEL}" \
     --n_heads "${N_HEADS}" \
     --n_layers "${N_LAYERS}" \
     --d_mlp 64 \
     --dropout 0.5 \
     --max_grad_norm 5.0 \
     --do_lower 0 \
     --replace_numbers 1 \
     --glove_embeddings "data/glove.840B.300d.txt" \
     --do_glove "${DO_GLOVE}" \
     --pool_outputs 0 \
     --seed "${SEED}" \
     --output_dir "output/conll_ner/standard_transformer/dmodel${D_MODEL}nheads${N_CAT}nlayers${N_LAYERS}/s${SEED}";
