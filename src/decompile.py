import argparse
import copy
from copy import deepcopy
from functools import partial
import itertools
import json
import math
from pathlib import Path
import random

import einops
import numpy as np
import pandas as pd
import re
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.models.programs import TransformerProgramModel, argmax

from src.run import set_seed, get_sample_fn
from src.utils import code_utils, data_utils, logging


logger = logging.get_logger(__name__)


class DotDict(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default="scratch")
    parser.add_argument("--output_dir", type=str, default="scratch")
    return parser.parse_args()


def load_model(path):
    args_fn = Path(path) / "args.json"
    if not args_fn.exists():
        raise ValueError(f"missing {args_fn}")
    model_fn = Path(path) / "model.pt"
    if not model_fn.exists():
        raise ValueError(f"missing {model_fn}")

    with open(args_fn, "r") as f:
        args = DotDict(json.load(f))

    logger.info(f"loading {args.dataset}")
    set_seed(args.seed)
    (
        train,
        test,
        val,
        idx_w,
        w_idx,
        idx_t,
        t_idx,
        X_train,
        Y_train,
        X_test,
        Y_test,
        X_val,
        Y_val,
    ) = data_utils.get_dataset(
        name=args.dataset,
        vocab_size=args.vocab_size,
        dataset_size=args.dataset_size,
        min_length=args.min_length,
        max_length=args.max_length,
        seed=args.seed,
        do_lower=args.do_lower,
        replace_numbers=args.replace_numbers,
        get_val=True,
        unique=args.unique,
    )

    logger.info(f"initializing model from {args_fn}")
    if args.d_var is None:
        d = max(len(idx_w), X_train.shape[-1])
    else:
        d = args.d_var
    init_emb = None
    if args.glove_embeddings and args.do_glove:
        emb = data_utils.get_glove_embeddings(
            idx_w, args.glove_embeddings, dim=args.n_vars_cat * d
        )
        init_emb = torch.tensor(emb, dtype=torch.float32).T
    unembed_mask = None
    if args.unembed_mask:
        unembed_mask = np.array([t in ("<unk>", "<pad>") for t in idx_t])
    set_seed(args.seed)
    model = TransformerProgramModel(
        d_vocab=len(idx_w),
        d_vocab_out=len(idx_t),
        n_vars_cat=args.n_vars_cat,
        n_vars_num=args.n_vars_num,
        d_var=d,
        n_heads_cat=args.n_heads_cat,
        n_heads_num=args.n_heads_num,
        d_mlp=args.d_mlp,
        n_cat_mlps=args.n_cat_mlps,
        n_num_mlps=args.n_num_mlps,
        mlp_vars_in=args.mlp_vars_in,
        n_layers=args.n_layers,
        n_ctx=X_train.shape[1],
        sample_fn=get_sample_fn(args.sample_fn),
        init_emb=init_emb,
        attention_type=args.attention_type,
        rel_pos_bias=args.rel_pos_bias,
        unembed_mask=unembed_mask,
        pool_outputs=args.pool_outputs,
        one_hot_embed=args.one_hot_embed,
        count_only=args.count_only,
        selector_width=args.selector_width,
    )

    logger.info(f"restoring weights from {model_fn}")
    model.load_state_dict(
        torch.load(str(model_fn), map_location=torch.device("cpu"))
    )

    model.set_temp(args.tau_end, argmax)
    return model, args, idx_w, idx_t, X_val


def model_to_code(model, args, idx_w, idx_t, X=None, output_dir=""):
    if output_dir:
        Path(output_dir).mkdir(exist_ok=True, parents=True)
    x = None
    if X is not None:
        x = idx_w[X[0]]
        x = x[x != "<pad>"].tolist()
    m = code_utils.model_to_code(
        model=model,
        idx_w=idx_w,
        idx_t=idx_t,
        embed_csv=not args.one_hot_embed,
        unembed_csv=True,
        one_hot=args.one_hot_embed,
        autoregressive=args.autoregressive,
        var_types=True,
        output_dir=output_dir,
        name=args.dataset,
        example=x,
        save=bool(output_dir),
    )
    return m


if __name__ == "__main__":
    args = parse_args()
    model, args, idx_w, idx_t, X_val = load_model(args.path)
    model_to_code(
        model=model,
        args=args,
        idx_w=idx_w,
        idx_t=idx_t,
        X=X_val,
        output_dir=args.output_dir,
    )
