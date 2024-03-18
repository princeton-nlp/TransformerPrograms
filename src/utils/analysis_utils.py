import collections
from collections import Counter
import itertools
import math
from pathlib import Path

from black import format_str, FileMode
import einops
import numpy as np
import pandas as pd
from pprint import pprint, pformat
import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader

from src.utils.code_utils import (
    cat_head_to_code,
    num_head_to_code,
    cat_mlp_to_code,
    num_mlp_to_code,
)


def get_var_tables(emb, idx_w, one_hot=True):
    with torch.no_grad():
        W_E = emb.get_W().cpu().T
    n_vars, var_dim = emb.n_vars, emb.d_var
    W = W_E.view(W_E.shape[0], n_vars, var_dim)
    lst = []
    var_to_toks = {}
    tok_to_val = {}
    for var in range(n_vars):
        if one_hot and var == 0:
            lst.append(f"# one_hot encoding")
            k = f"one_hot"
        else:
            lst.append(f"# var{var}_embeddings")
            k = f"var{var}_embeddings"
        var_to_toks[k] = {}
        tok_to_val[k] = {}
        v_hat = W[:, var].argmax(-1)
        for val in range(var_dim):
            mask = v_hat == val
            # wds = ", ".join([f"'{w}'" for w in idx_w[mask]])
            wds = set(idx_w[mask])
            var_to_toks[k][val] = wds
            for w in wds:
                tok_to_val[k][w] = val
    return tok_to_val, var_to_toks


def get_var_types(model, idx_w, cat_var_names=None, one_hot=False):
    if cat_var_names is None:
        cat_var_names, _, _ = get_var_names(model, idx_w=idx_w, one_hot=one_hot)
    d = {}
    if one_hot:
        d["tokens"] = idx_w.tolist()
        if len(idx_w) < model.d_var:
            d["tokens"] += [""] * (model.d_var - len(idx_w))
    else:
        for i in range(model.n_vars_cat):
            d[f"var{i}_embeddings"] = [
                f"Var{i}._{j}" for j in range(model.d_var)
            ]
    d["positions"] = list(range(model.d_var))
    for l, block in enumerate(model.blocks):
        attn = block.cat_attn
        pi_V = attn.W_V.get_W().detach().cpu()
        val_names = cat_var_names[pi_V.argmax(-1)]
        for h, v in enumerate(val_names):
            if v in d:
                d[f"attn_{l}_{h}_outputs"] = d[v]
            else:
                # print(v)
                continue
        attn = block.cat_attn
        pi_V = attn.W_V.get_W().detach().cpu()
        val_names = cat_var_names[pi_V.argmax(-1)]
    return d


def get_type(s):
    for k in ("attn", "mlp"):
        if k in s:
            return k
    return s


def get_layer(s):
    parts = s.split("_")
    if len(parts) < 2:
        return 0
    return int(parts[-2]) + 1


def analyze_attn_head(
    model,
    cat_var_names=None,
    head=0,
    layer=0,
    autoregressive=False,
    idx_w=None,
    var_types=None,
    one_hot=False,
    attn_type="cat",
):
    if cat_var_names is None:
        cat_var_names, num_var_names, all_var_names = get_var_names(
            model, idx_w=idx_w, one_hot=one_hot
        )
    if attn_type == "cat":
        attn = model.blocks[layer].cat_attn
    else:
        attn = model.blocks[layer].num_attn
    W_K, W_Q, W_V = [
        W.detach().cpu() for W in (attn.W_K(), attn.W_Q(), attn.W_V())
    ]
    W_pred = attn.W_pred.get_W().detach().cpu()
    pi_K, pi_Q, pi_V = [
        f.get_W().detach().cpu() for f in (attn.W_K, attn.W_Q, attn.W_V)
    ]
    n_heads = pi_K.shape[0]
    n_vars = pi_K.shape[1]
    var_dim = W_K.shape[-1] // n_vars
    key_names = cat_var_names[pi_K.argmax(-1)]
    query_names = cat_var_names[pi_Q.argmax(-1)]
    if attn_type == "cat":
        val_names = cat_var_names[pi_V.argmax(-1)]
    else:
        val_names = num_var_names[pi_V.argmax(-1)]
    if n_heads == 1:
        key_names, query_names, val_names = (
            [key_names],
            [query_names],
            [val_names],
        )
    h = head
    q, k, v = query_names[h], key_names[h], val_names[h]
    W_pred = W_pred[h]
    header = []
    q_name, k_name = f"{q[:-1]}", f"{k[:-1]}"
    if q_name == k_name:
        q_name, k_name = f"q_{q_name}", f"k_{k_name}"
    header.append(f"def predicate_{layer}_{head}({q_name}, {k_name}):")
    out = []
    stmts = collections.defaultdict(list)
    for q_i in range(W_pred.shape[0]):
        k_j = (W_pred[q_i]).argmax(-1).item()
        if var_types is not None and k in var_types:
            k_j = var_types[k][k_j]
            if type(k_j) == str:
                k_j = f"'{k_j}'"
        if var_types is not None and q in var_types:
            q_i = var_types[q][q_i]
        if q_i in ("", "<pad>"):  # or k_j in ("", "'<pad>'"):
            continue
        stmt = f"\treturn {k_name} == {k_j}"
        stmts[stmt].append(q_i)
    for i, (stmt, q_is) in enumerate(stmts.items()):
        cond = "if" if i == 0 else "elif"
        out.append(f"{cond} {q_name} in {set(q_is)}:")
        out += [stmt]

    if attn_type == "cat":
        head_name = f"attn_{layer}_{head}"
    else:
        head_name = f"num_attn_{layer}_{head}"

    df = pd.DataFrame(
        {
            "name": [head_name],
            "type": attn_type,
            "layer": layer + 1,
            "query": q,
            "key": k,
            "value": v,
            "query_type": get_type(q),
            "key_type": get_type(k),
            "value_type": get_type(v),
            "query_layer": get_layer(q),
            "key_layer": get_layer(k),
            "value_layer": get_layer(v),
            "max_statements": W_pred.shape[0],
            "statements": len(stmts),
            "max_lines": 1 + W_pred.shape[0] * 2,
            "lines": 1 + len(stmts) * 2,
        }
    )
    return df


def analyze_cat_mlp(
    model, layer=0, n_mlp=0, var_names=None, return_df=False, var_types=None
):
    mlp = model.blocks[layer].cat_mlp.mlps[n_mlp]
    n_vars = mlp.W_read.n_vars
    if var_names is None:
        var_names, _, _ = get_var_names(model)
    mlp.eval()
    read = mlp.W_read
    with torch.no_grad():
        vars_in = torch.argmax(read.W, dim=-1).cpu().numpy()
    mlp_vars_in, n_vars = read.W.shape
    var_dims = [mlp.d_out for _ in range(mlp_vars_in)]
    input_idxs = np.array(
        list(itertools.product(*[range(d) for d in var_dims]))
    )
    X = np.zeros((len(input_idxs), read.d_in), dtype=np.float32)
    l = np.arange(X.shape[0])
    for i, j in enumerate(vars_in):
        X[l, input_idxs[:, i] + (var_dims[i] * j)] = 1
    X = torch.tensor(X, device=mlp.W_in.device)
    with torch.no_grad():
        mlp_out = mlp(X.unsqueeze(1)).squeeze(1).detach().cpu()
    mlp_var_out = mlp_out.argmax(-1).numpy()
    order = np.argsort(mlp_var_out)
    mlp_var_out = mlp_var_out[order]
    input_idxs = input_idxs[order]
    max_keys = input_idxs.shape[0]
    mlp_var_names = var_names[vars_in]
    assert len(mlp_var_names) == 2
    var1, var2 = mlp_var_names
    lst = []
    if len(set(mlp_var_names)) == 1:
        m = input_idxs[:, 0] == input_idxs[:, 1]
        input_idxs = input_idxs[m, :1]
        mlp_var_out = mlp_var_out[m]
        mlp_var_names = mlp_var_names[:1]
    res_name = f"mlp{layer}[i]"
    out = []
    mlp_ks = [v[:-1] for v in mlp_var_names]
    if var_types is not None:
        for j, v in enumerate(mlp_var_names):
            if v in var_types:
                vs = var_types[v]
                lst.append([vs[i] for i in input_idxs[:, j]])
            else:
                lst.append(input_idxs[:, j].tolist())
        input_idxs = list(zip(*lst))
    counts = Counter(mlp_var_out.tolist()).most_common()
    var_out_to_keys = {}
    for i, (var_out, _) in enumerate(counts[1:]):
        m = mlp_var_out == var_out
        keys = [row for b, row in zip(m, input_idxs) if b]
        var_out_to_keys[var_out] = keys
        if len(mlp_ks) == 1:
            ks = set([row[0] for row in keys])
        else:
            ks = set(
                [
                    tuple(row)
                    for row in keys
                    if not ("" in tuple(row) or "<pad>" in tuple(row))
                ]
            )
        if len(ks):
            var_out_to_keys[var_out] = ks

    df = pd.DataFrame(
        {
            "name": [f"mlp_{layer}_{n_mlp}"],
            "type": "cat_mlp",
            "layer": layer + 1,
            "var1": var1,
            "var2": var2,
            "var1_type": get_type(var1),
            "var2_type": get_type(var2),
            "var1_layer": get_layer(var1),
            "var2_layer": get_layer(var2),
            "max_keys": max_keys,
            "keys": sum(len(ks) for ks in var_out_to_keys.values()),
            "max_branches": mlp.d_out,
            "branches": len(var_out_to_keys),
        }
    )
    return df


def analyze_num_mlp(
    model,
    layer=0,
    n_mlp=0,
    var_names=None,
    max_n=None,
    return_df=False,
    var_types=None,
):
    mlp = model.blocks[layer].num_mlp.mlps[n_mlp]
    if max_n is None:
        max_n = model.pos_embed.max_ctx * (layer + 1)
    n_vars = mlp.W_read.n_vars
    if var_names is None:
        _, var_names, _ = get_var_names(model)
    mlp.eval()
    read = mlp.W_read
    with torch.no_grad():
        vars_in = torch.argmax(read.W, dim=-1).cpu().numpy()
    mlp_vars_in, n_vars = read.W.shape
    var_dims = [max_n for _ in range(mlp_vars_in)]
    input_idxs = np.array(
        list(itertools.product(*[range(d) for d in var_dims]))
    )
    X = np.zeros((len(input_idxs), read.d_in), dtype=np.float32)
    l = np.arange(X.shape[0])
    for i, j in enumerate(vars_in):
        X[l, j] = input_idxs[:, i]
    X = torch.tensor(X, device=mlp.W_in.device)
    with torch.no_grad():
        mlp_out = mlp(X.unsqueeze(1)).squeeze(1).detach().cpu()
    mlp_var_out = mlp_out.argmax(-1).numpy()
    mlp_var_names = var_names[vars_in]
    # out = [f"# num_mlp_{layer}_{head}: {mlp_var_names}"]
    max_keys = input_idxs.shape[0]
    assert len(mlp_var_names) == 2
    var1, var2 = mlp_var_names
    if len(set(mlp_var_names)) == 1:
        m = input_idxs[:, 0] == input_idxs[:, 1]
        input_idxs = input_idxs[m, :1]
        mlp_var_out = mlp_var_out[m]
        mlp_var_names = mlp_var_names[:1]
    mlp_ks = [v[:-1] for v in mlp_var_names]
    counts = Counter(mlp_var_out.tolist()).most_common()
    var_out_to_keys = {}
    for i, (var_out, _) in enumerate(counts[1:]):
        m = mlp_var_out == var_out
        keys = input_idxs[m]
        if len(mlp_ks) == 1:
            ks = sorted([row[0] for row in keys])
        else:
            ks = sorted([tuple(row) for row in keys])
        if len(ks):
            var_out_to_keys[var_out] = ks
    df = pd.DataFrame(
        {
            "name": [f"num_mlp_{layer}_{n_mlp}"],
            "type": "num_mlp",
            "layer": layer + 1,
            "var1": var1,
            "var2": var2,
            "var1_type": get_type(var1),
            "var2_type": get_type(var2),
            "var1_layer": get_layer(var1),
            "var2_layer": get_layer(var2),
            "max_keys": max_keys,
            "keys": sum(len(ks) for ks in var_out_to_keys.values()),
            "max_branches": mlp.d_out,
            "branches": len(var_out_to_keys),
        }
    )
    return df


def annotate(model, idx_w=None, one_hot=True):
    if one_hot:
        k = 3
        if idx_w is None:
            idx_w = [str(i) for i in range(model.d_var)]
        else:
            idx_w = [
                w.replace("<s>", "bos")
                .replace("<pad>", "pad")
                .replace("</s>", "eos")
                .replace("<unk>", "unk")
                for w in idx_w
            ]
        cat_val_names = [f"tokens: {w:>{k}}" for w in idx_w]
        if model.d_pos > len(idx_w):
            cat_val_names += [
                f"tokens: {'':>{k}}" for _ in range(model.d_pos - len(idx_w))
            ]
        cat_val_names += [
            f"var{i}_embeddings: {j:>{k}}"
            for i in range(1, model.n_vars_cat)
            for j in range(model.d_var)
        ]
    else:
        k = 2
        cat_val_names = [
            f"var{i}_embeddings: {j:>{k}}"
            for i in range(model.n_vars_cat)
            for j in range(model.d_var)
        ]
    cat_val_names += [f"positions: {j:>{k}}" for j in range(model.d_pos)]
    for layer in range(model.num_layers):
        cat_val_names += [
            # f"head{layer}_{i}: {j:>{k}}"
            f"attn_{layer}_{i}_outputs: {j:>{k}}"
            for i in range(model.n_heads_cat)
            for j in range(model.d_var)
        ]
        cat_val_names += [
            # f"mlp{layer}_{i}: {j:>{k}}"
            f"mlp_{layer}_{i}_outputs: {j:>{k}}"
            for i in range(model.n_cat_mlps)
            for j in range(model.d_var)
        ]
        cat_val_names += [
            f"num_mlp_{layer}_{i}_outputs: {j:>{k}}"
            for i in range(model.n_num_mlps)
            for j in range(model.d_var)
        ]

    cat_val_names = np.array(cat_val_names)

    if torch.all(model.num_embed.W_E == 1).item():
        num_val_names = [f"ones: {' '*k}" for i in range(model.n_vars_num)]
    else:
        num_val_names = [
            f"num_var{i}_embeddings: {' '*k}" for i in range(model.n_vars_num)
        ]
    for layer in range(model.num_layers):
        num_val_names += [
            f"num_attn_{layer}_{i}_outputs:" for i in range(model.n_heads_num)
        ]
        # num_val_names += [f"nmlp_{layer}_{i}:" for i in range(model.n_num_mlps)]
    num_val_names = np.array(num_val_names)
    all_val_names = np.concatenate([cat_val_names, num_val_names])
    return cat_val_names, num_val_names, all_val_names


def get_var_names(model, idx_w=None, one_hot=True):
    if one_hot:
        cat_var_names = [f"tokens"] + [
            f"var{i}_embeddings" for i in range(model.n_vars_cat - 1)
        ]
    else:
        cat_var_names = [f"var{i}_embeddings" for i in range(model.n_vars_cat)]
    cat_var_names += [f"positions"]
    for layer in range(model.num_layers):
        cat_var_names += [
            # f"head{layer}_{i}" for i in range(model.n_heads_cat)
            f"attn_{layer}_{i}_outputs"
            for i in range(model.n_heads_cat)
        ]
        # cat_var_names += [f"mlp{layer}_{i}" for i in range(model.n_cat_mlps)]
        cat_var_names += [
            f"mlp_{layer}_{i}_outputs" for i in range(model.n_cat_mlps)
        ]
        cat_var_names += [
            f"num_mlp_{layer}_{i}_outputs"
            for i in range(model.n_num_mlps)
            # f"nc_mlp{layer}_{i}" for i in range(model.n_num_mlps)
        ]
    cat_var_names = np.array(cat_var_names)

    if torch.all(model.num_embed.W_E == 1).item():
        num_var_names = [f"ones" for i in range(model.n_vars_num)]
    else:
        num_var_names = [
            f"num_var{i}_embeddings" for i in range(model.n_vars_num)
        ]
    for layer in range(model.num_layers):
        num_var_names += [
            # f"nhead{layer}_{i}" for i in range(model.n_heads_num)
            f"num_attn_{layer}_{i}_outputs"
            for i in range(model.n_heads_num)
        ]
        # num_var_names += [f"nmlp_l{layer}h{i}" for i in range(model.n_num_mlps)]
    num_var_names = np.array(num_var_names)

    all_var_names = np.concatenate([cat_var_names, num_var_names])

    return cat_var_names, num_var_names, all_var_names


def analyze_model(
    model,
    idx_w,
    idx_t,
    one_hot=False,
    autoregressive=False,
    var_types=None,
):
    if var_types == True:
        var_types = get_var_types(model, idx_w, one_hot=one_hot)
    rows = []
    num_layers = len(model.blocks)
    for l in range(num_layers):
        for h in range(model.blocks[0].n_heads_cat):
            rows.append(
                analyze_attn_head(
                    model,
                    layer=l,
                    head=h,
                    autoregressive=autoregressive,
                    idx_w=idx_w if one_hot else None,
                    var_types=var_types,
                    one_hot=one_hot,
                    attn_type="cat",
                )
            )
        for h in range(model.blocks[0].n_heads_num):
            rows.append(
                analyze_attn_head(
                    model,
                    layer=l,
                    head=h,
                    autoregressive=autoregressive,
                    idx_w=idx_w if one_hot else None,
                    var_types=var_types,
                    one_hot=one_hot,
                    attn_type="num",
                )
            )
        block = model.blocks[l]
        for h in range(block.n_cat_mlps):
            rows.append(
                analyze_cat_mlp(model, layer=l, n_mlp=h, var_types=var_types)
            )
        for h in range(block.n_num_mlps):
            rows.append(
                analyze_num_mlp(model, layer=l, n_mlp=h, var_types=var_types)
            )
    return pd.concat(rows)


def fmtb(s):
    return format_str(s, mode=FileMode())


def analyze_loc(
    model,
    idx_w,
    idx_t,
    one_hot=False,
    autoregressive=False,
    var_types=None,
):
    if var_types == True:
        var_types = get_var_types(model, idx_w, one_hot=one_hot)
    rows = []
    full_rows = []
    num_layers = len(model.blocks)
    for l in range(num_layers):
        for h in range(model.blocks[0].n_heads_cat):
            for compress in (True, False):
                s = cat_head_to_code(
                    model,
                    layer=l,
                    head=h,
                    autoregressive=autoregressive,
                    idx_w=idx_w if one_hot else None,
                    var_types=var_types,
                    one_hot=one_hot,
                    compress=compress,
                )
                rows.append(
                    {
                        "name": [f"attn_{l}_{h}"],
                        "type": "cat_attn",
                        "compress": compress,
                        "lines": len(fmtb(s).split("\n")),
                        "code": fmtb(s),
                    }
                )
        for h in range(model.blocks[0].n_heads_num):
            for compress in (True, False):
                s = num_head_to_code(
                    model,
                    layer=l,
                    head=h,
                    autoregressive=autoregressive,
                    idx_w=idx_w if one_hot else None,
                    var_types=var_types,
                    one_hot=one_hot,
                    compress=compress,
                )
                rows.append(
                    {
                        "name": [f"num_attn_{l}_{h}"],
                        "type": "num_attn",
                        "compress": compress,
                        "lines": len(fmtb(s).split("\n")),
                        "code": fmtb(s),
                    }
                )
        block = model.blocks[l]
        for h in range(block.n_cat_mlps):
            for compress in (True, False):
                s = cat_mlp_to_code(
                    model,
                    layer=l,
                    n_mlp=h,
                    var_types=var_types,
                    compress=compress,
                )
                rows.append(
                    {
                        "name": [f"mlp_{l}_{h}"],
                        "type": "cat_mlp",
                        "compress": compress,
                        "lines": len(fmtb(s).split("\n")),
                        "code": fmtb(s),
                    }
                )
        for h in range(block.n_num_mlps):
            for compress in (True, False):
                s = num_mlp_to_code(
                    model,
                    layer=l,
                    n_mlp=h,
                    var_types=var_types,
                    compress=compress,
                )
                rows.append(
                    {
                        "name": [f"num_mlp_{l}_{h}"],
                        "type": "num_mlp",
                        "compress": compress,
                        "lines": len(fmtb(s).split("\n")),
                        "code": fmtb(s),
                    }
                )
    return pd.concat([pd.DataFrame(r) for r in rows])


def fmt(lst):
    return [w.replace("<pad>", "pad").replace("<s>", "bos") for w in lst]
