import collections
from collections import Counter
import itertools
import math
from pathlib import Path
import re

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


def embed_to_code(emb, idx_w, one_hot=True, var_types=None):
    with torch.no_grad():
        W_E = emb.get_W().cpu().T
    n_vars, var_dim = emb.n_vars, emb.d_var
    W = W_E.view(W_E.shape[0], n_vars, var_dim)
    tok_to_val, _ = get_var_tables(emb, idx_w, one_hot=one_hot)
    lst = []
    for var, d in tok_to_val.items():
        if var == "one_hot" and var_types is not None:
            continue
        lst.append(
            f"w_to_{var} = {d}" if var != "one_hot" else f"one_hot = {d}"
        )
    for i, var in enumerate(tok_to_val.keys()):
        if one_hot and i == 0:
            if var_types is None:
                lst.append(f"tokens = [one_hot[w] for w in x]")
        else:
            lst.append(f"var{i}_embeddings = [w_to_{var}[w] for w in x]")
    return "\n".join(lst)


def get_var_tables(emb, idx_w, one_hot=True, do_set=True):
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
            if do_set:
                wds = set(idx_w[mask])
            else:
                wds = idx_w[mask].tolist()
            var_to_toks[k][val] = wds
            for w in wds:
                tok_to_val[k][w] = val
    return tok_to_val, var_to_toks


def get_embed_df(emb, idx_w):
    tok_to_val, var_to_toks = get_var_tables(emb, idx_w, one_hot=False)
    tok_df = {"word": idx_w}
    for var in tok_to_val.keys():
        tok_df[var] = [tok_to_val[var][w] for w in idx_w]
    df = pd.DataFrame(tok_df)
    return df.set_index("word")


def get_select_closest(autoregressive=False):
    if autoregressive:
        line = "matches = [j for j, k in enumerate(keys[:i + 1]) if predicate(q, k)]"
    else:
        line = "matches = [j for j, k in enumerate(keys) if predicate(q, k)]"
    return [
        "def select_closest(keys, queries, predicate):",
        "\tscores = [[False for _ in keys] for _ in queries]",
        "\tfor i, q in enumerate(queries):",
        ("\t\t" + line),
        "\t\tif not(any(matches)):",
        "\t\t\tscores[i][0] = True",
        "\t\telse:",
        "\t\t\tj = min(matches, key=lambda j: len(matches) if j == i else abs(i - j))",
        "\t\t\tscores[i][j] = True",
        "\treturn scores",
    ]


def get_select(autoregressive=False):
    if autoregressive:
        line = (
            "\treturn [[predicate(q, k) for k in keys[:i+1]] "
            "for i, q in enumerate(queries)]"
        )
    else:
        line = "\treturn [[predicate(q, k) for k in keys] for q in queries]"
    return ["def select(keys, queries, predicate):", line]


def get_aggregate():
    return [
        "def aggregate(attention, values):",
        "\treturn [[v for a, v in zip(attn, values) if a][0] for attn in attention]",
    ]


def get_aggregate_sum():
    return [
        "def aggregate_sum(attention, values):",
        "\treturn [sum([v for a, v in zip(attn, values) if a]) for attn in attention]",
    ]


def get_var_types(model, idx_w, cat_var_names=None, one_hot=False, enums=True):
    if cat_var_names is None:
        cat_var_names, _, _ = get_var_names(model, idx_w=idx_w, one_hot=one_hot)
    d = {}
    if one_hot:
        d["tokens"] = idx_w.tolist()
        if len(idx_w) < model.d_var:
            d["tokens"] += [""] * (model.d_var - len(idx_w))
    elif enums:
        for i in range(model.n_vars_cat):
            d[f"var{i}_embeddings"] = [
                f"Emb{i}.V{j:02}" for j in range(model.d_var)
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
                continue
        attn = block.cat_attn
        pi_V = attn.W_V.get_W().detach().cpu()
        val_names = cat_var_names[pi_V.argmax(-1)]
    return d


def cat_head_to_code(
    model,
    cat_var_names=None,
    head=0,
    layer=0,
    autoregressive=False,
    idx_w=None,
    var_types=None,
    one_hot=False,
    compress=True,
):
    if cat_var_names is None:
        cat_var_names, num_var_names, all_var_names = get_var_names(
            model, idx_w=idx_w, one_hot=one_hot
        )
    attn = model.blocks[layer].cat_attn
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
    val_names = cat_var_names[pi_V.argmax(-1)]
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
        if q_i in ("", "<pad>") and compress:
            continue
        stmt = f"\treturn {k_name} == {k_j}"
        if not compress:
            stmt = f"{stmt}  # {q_i}"
        stmts[stmt].append(q_i)
    for i, (stmt, q_is) in enumerate(stmts.items()):
        cond = "if" if i == 0 else "elif"
        out.append(f"{cond} {q_name} in {set(q_is)}:")
        out += [stmt]
    lst = header + ["\t" + s for s in out]
    lst.append(
        f"attn_{layer}_{head}_pattern = "
        f"select_closest({k}, {q}, predicate_{layer}_{head})"
    )
    lst.append(
        f"attn_{layer}_{head}_outputs = aggregate(attn_{layer}_{head}_pattern, {v})"
    )
    return "\n".join(lst)


def num_head_to_code(
    model,
    cat_var_names=None,
    num_var_names=None,
    head=0,
    layer=0,
    autoregressive=False,
    idx_w=None,
    var_types=None,
    one_hot=False,
    compress=True,
):
    if cat_var_names is None:
        cat_var_names, _, _ = get_var_names(model, idx_w=idx_w, one_hot=one_hot)
    if num_var_names is None:
        _, num_var_names, _ = get_var_names(model, idx_w=idx_w, one_hot=one_hot)
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
    header.append(f"def num_predicate_{layer}_{head}({q_name}, {k_name}):")
    out = []
    stmts = collections.defaultdict(list)
    for q_i in range(W_pred.shape[0]):
        k_j = (W_pred[q_i]).argmax(-1).item()
        if var_types is not None:
            if k in var_types:
                k_j = var_types[k][k_j]
                if type(k_j) == str:
                    k_j = f"'{k_j}'"
            if q in var_types:
                q_i = var_types[q][q_i]
            if (k_j in ("", "<pad>") or q_i in ("", "<pad>")) and compress:
                continue
        stmt = f"\treturn {k_name} == {k_j}"
        if not compress:
            stmt = f"{stmt}  # {q_i}"
        stmts[stmt].append(q_i)
    for i, (stmt, q_is) in enumerate(stmts.items()):
        cond = "if" if i == 0 else "elif"
        out.append(f"{cond} {q_name} in {set(q_is)}:")
        out += [stmt]
    lst = header + ["\t" + s for s in out]
    lst.append(
        f"num_attn_{layer}_{head}_pattern = "
        f"select({k}, {q}, num_predicate_{layer}_{head})"
    )
    lst.append(
        (
            f"num_attn_{layer}_{head}_outputs ="
            f" aggregate_sum(num_attn_{layer}_{head}_pattern, {v})"
        )
    )
    return "\n".join(lst)


def cat_mlp_to_code(
    model,
    layer=0,
    n_mlp=0,
    var_names=None,
    return_df=False,
    var_types=None,
    compress=True,
    one_hot=True,
):
    mlp = model.blocks[layer].cat_mlp.mlps[n_mlp]
    n_vars = mlp.W_read.n_vars
    if var_names is None:
        var_names, _, _ = get_var_names(model, one_hot=one_hot)
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
    mlp_var_names = var_names[vars_in]
    lst = []
    if len(set(mlp_var_names)) == 1 and compress:
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

    out += [f"# mlp_{layer}_{n_mlp} " + "#" * 53]
    out.append(f"def mlp_{layer}_{n_mlp}({', '.join(mlp_ks)}):")
    out += [f"\tkey = ({', '.join(mlp_ks)})"]
    cond_i = 0
    for i, (var_out, _) in enumerate(counts[int(compress) :]):
        m = mlp_var_out == var_out
        keys = [row for b, row in zip(m, input_idxs) if b]
        if len(mlp_ks) == 1:
            s = "{" + str(sorted(set([row[0] for row in keys])))[1:-1] + "}"
        elif compress:
            rows = [
                tuple(row)
                for row in keys
                if not ("" in tuple(row) or "<pad>" in tuple(row))
            ]
            ks = sorted(set(rows))
            if not ks:
                continue
            s = "{" + str(ks)[1:-1] + "}"
        else:
            ks = sorted([tuple(row) for row in keys])
            s = str(ks)
        cond = "if" if cond_i == 0 else "elif"
        out += [f"\t{cond} key in {s}:"]
        out += [f"\t\treturn {var_out}"]
        cond_i += 1
    if compress:
        out.append(f"\treturn {counts[0][0]}")
    k = (
        f"zip({', '.join(mlp_var_names.tolist())})"
        if len(mlp_var_names) > 1
        else f" {mlp_var_names[0]}"
    )
    ks = ", ".join([f"k{i}" for i in range(len(mlp_var_names))])
    out.append(
        f"mlp_{layer}_{n_mlp}_outputs = "
        f"[mlp_{layer}_{n_mlp}({ks}) for {ks} in {k}]"
    )
    return "\n".join(out)


def num_mlp_to_code(
    model,
    layer=0,
    n_mlp=0,
    var_names=None,
    max_n=None,
    return_df=False,
    var_types=None,
    compress=True,
    one_hot=True,
):
    mlp = model.blocks[layer].num_mlp.mlps[n_mlp]
    if max_n is None:
        max_n = model.pos_embed.max_ctx * (layer + 1)
    n_vars = mlp.W_read.n_vars
    if var_names is None:
        _, var_names, _ = get_var_names(model, one_hot=one_hot)
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
    out = []
    if compress and len(set(mlp_var_names)) == 1:
        m = input_idxs[:, 0] == input_idxs[:, 1]
        input_idxs = input_idxs[m, :1]
        mlp_var_out = mlp_var_out[m]
        mlp_var_names = mlp_var_names[:1]
    mlp_ks = [v[:-1] for v in mlp_var_names]
    out += [f"# num_mlp_{layer}_{n_mlp} " + "#" * 49]
    out.append(f"def num_mlp_{layer}_{n_mlp}({', '.join(mlp_ks)}):")
    counts = Counter(mlp_var_out.tolist()).most_common()
    out += [f"\tkey = ({', '.join(mlp_ks)})"]
    cond_i = 0
    for i, (var_out, _) in enumerate(counts[int(compress) :]):
        m = mlp_var_out == var_out
        keys = [row for b, row in zip(m, input_idxs) if b]
        if len(mlp_ks) == 1:
            s = "{" + str(sorted(set([row[0] for row in keys])))[1:-1] + "}"
        elif compress:
            ks = sorted(set([tuple(row) for row in keys]))
            if not ks:
                continue
            s = "{" + str(ks)[1:-1] + "}"
        else:
            ks = sorted([tuple(row) for row in keys])
            s = str(ks)
        cond = "if" if cond_i == 0 else "elif"
        out += [f"\t{cond} key in {s}:"]
        out += [f"\t\treturn {var_out}"]
        cond_i += 1
    if compress:
        out.append(f"\treturn {counts[0][0]}")
    k = (
        f"zip({', '.join(mlp_var_names.tolist())})"
        if len(mlp_var_names) > 1
        else f" {mlp_var_names[0]}"
    )
    ks = ", ".join([f"k{i}" for i in range(len(mlp_var_names))])
    out.append(
        f"num_mlp_{layer}_{n_mlp}_outputs = "
        f"[num_mlp_{layer}_{n_mlp}({ks}) for {ks} in {k}]"
    )
    return "\n".join(out)


def get_unembed_df(
    model,
    idx_t,
    val_names=None,
    var_types=None,
    one_hot=True,
    enums=False,
    unembed_mask=True,
):
    W_U = model.unembed.W_U.detach().cpu().numpy()
    if unembed_mask:
        m = ~np.array([t in ("<unk>", "<pad>") for t in idx_t])
        idx_t = idx_t[m]
        W_U = W_U[:, m]

    if val_names is None:
        _, _, val_names = get_value_names(model, one_hot=one_hot, enums=enums)
    var_names = list(set([v.split(":")[0] for v in val_names]))
    vals = []
    for v in val_names:
        var_name = v.split(":")[0]
        val = v.split(":")[-1].strip()
        if val.isnumeric():
            val = int(val)
            if var_types is not None and var_name in var_types:
                val = var_types[var_name][val]
        if val == "":
            val = "_"
        vals.append(val)
    vals = np.array(vals)
    rows = []
    for var in var_names:
        m = np.array([val.startswith(var) for val in val_names])
        df = pd.DataFrame(W_U[m], columns=idx_t)
        df["feature"] = var
        df["value"] = vals[m]
        rows.append(df.set_index(["feature", "value"]))
    return pd.concat(rows).sort_index()


def unembed_df_to_file(model, idx_t, val_names=None, one_hot=True):
    W_U = model.unembed.W_U.detach().cpu().numpy()
    if val_names is None:
        _, _, val_names = get_value_names(model, one_hot=one_hot)
    var_names = list(set([v.split(":")[0] for v in val_names]))
    vals = []
    for v in val_names:
        val = v.split(":")[-1].strip()
        if val.isnumeric():
            val = int(val)
        vals.append(val)
    vals = np.array(vals)
    rows = ["import numpy as np", "\n"]
    d = {}
    for var in var_names:
        m = np.array([val.startswith(var) for val in val_names])
        d[var] = f"np.array({W_U[m].tolist()})"
    s = "{" + ",\n".join([f"'{k}': {v}" for k, v in d.items()]) + "}"
    rows += [f"weights = {s}"]
    rows += [f"def get_weights():"]
    rows += ["\treturn weights"]
    return "\n".join(rows)


def get_embed_enum(n_vars):
    return [
        "def embed_enum(cls, w):",
        "\tfor val in cls:",
        "\t\tif w in val.value:",
        "\t\t\treturn val",
        "\traise ValueError(w)",
    ]


def embed_enum_to_file(emb, idx_w):
    tok_to_val, var_to_toks = get_var_tables(
        emb, idx_w, one_hot=False, do_set=False
    )
    n_vars, var_dim = emb.n_vars, emb.d_var
    lst = [f"from enum import Enum"]
    for var, val_to_toks in var_to_toks.items():
        k = var.split("_")[0][3:]
        lst += ["\n", f"class Emb{k}(Enum):"]
        for val, toks in val_to_toks.items():
            lst += [f"\tV{val:02} = {toks}"]
    return "\n".join(lst)


def embed_enum_to_code(emb, fn="embeddings.csv"):
    lst = [f"embed_df = pd.read_csv('{fn}').set_index('word')"]
    lst += ["embeddings = embed_df.loc[tokens]"]
    k = r"f'V{i:02}'"
    lst += [
        f"var{j}_embeddings = [Emb{j}[{k}] for i in embeddings['var{j}_embeddings']]"
        for j in range(emb.n_vars)
    ]
    return "\n".join(lst)


def __embed_enum_to_code(emb, **kwargs):
    lst = [
        f"var{j}_embeddings = [embed_enum(Emb{j}, w) for w in tokens]"
        for j in range(emb.n_vars)
    ]
    return "\n".join(lst)


def embed_csv_to_code(emb, idx_w, fn="embed.csv"):
    n_vars, var_dim = emb.n_vars, emb.d_var
    lst = [f"embed_df = pd.read_csv('{fn}').set_index('word')"]
    lst += ["embeddings = embed_df.loc[tokens]"]
    lst += [
        f"var{i}_embeddings = embeddings['var{i}_embeddings'].tolist()"
        for i in range(n_vars)
    ]
    return "\n".join(lst)


def fmt(lst):
    return [
        w.replace("<pad>", "pad")
        .replace("<s>", "bos")
        .replace("</s>", "eos")
        .replace("<unk>", "unk")
        for w in lst
    ]


def get_value_names(model, idx_w=None, one_hot=True, enums=False):
    if one_hot:
        k = 3
        if idx_w is None:
            idx_w = [str(i) for i in range(model.d_var)]
        else:
            idx_w = fmt(idx_w)
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
    # elif enums:
    #     k = 2
    #     s = [
    #         f"Emb{i}.V{j}"
    #         for i in range(model.n_vars_cat)
    #         for j in range(model.d_var)
    #     ]
    #     cat_val_names = [f"var{i}_embeddings: {t:>{k}}" for t in s]
    else:
        k = 2
        cat_val_names = [
            f"var{i}_embeddings: {j:>{k}}"
            for i in range(model.n_vars_cat)
            for j in range(model.d_var)
        ]
    cat_val_names += [f"positions: {j:>{k}}" for j in range(model.d_pos)]
    for layer in range(model.n_layers):
        cat_val_names += [
            f"attn_{layer}_{i}_outputs: {j:>{k}}"
            for i in range(model.n_heads_cat)
            for j in range(model.d_var)
        ]
        cat_val_names += [
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
    for layer in range(model.n_layers):
        num_val_names += [
            f"num_attn_{layer}_{i}_outputs:" for i in range(model.n_heads_num)
        ]
    num_val_names = np.array(num_val_names)
    all_val_names = np.concatenate([cat_val_names, num_val_names])
    return cat_val_names, num_val_names, all_val_names


def get_var_names(model, idx_w=None, one_hot=True):
    if one_hot:
        cat_var_names = [f"tokens"]
    else:
        cat_var_names = [f"var{i}_embeddings" for i in range(model.n_vars_cat)]
    cat_var_names += [f"positions"]
    for layer in range(model.n_layers):
        cat_var_names += [
            f"attn_{layer}_{i}_outputs" for i in range(model.n_heads_cat)
        ]
        cat_var_names += [
            f"mlp_{layer}_{i}_outputs" for i in range(model.n_cat_mlps)
        ]
        cat_var_names += [
            f"num_mlp_{layer}_{i}_outputs" for i in range(model.n_num_mlps)
        ]
    cat_var_names = np.array(cat_var_names)

    if torch.all(model.num_embed.W_E == 1).item():
        num_var_names = [f"ones" for i in range(model.n_vars_num)]
    else:
        num_var_names = [
            f"num_var{i}_embeddings" for i in range(model.n_vars_num)
        ]
    for layer in range(model.n_layers):
        num_var_names += [
            f"num_attn_{layer}_{i}_outputs" for i in range(model.n_heads_num)
        ]
    num_var_names = np.array(num_var_names)

    all_var_names = np.concatenate([cat_var_names, num_var_names])

    return cat_var_names, num_var_names, all_var_names


def get_score(var_name):
    if var_name.startswith("num_attn") or var_name == "ones":
        return (
            f"{var_name[:-1]}_scores = classifier_weights.loc["
            f"[('{var_name}', '_') for v in {var_name}]]"
            f".mul({var_name}, axis=0)"
        )
    else:
        return (
            f"{var_name[:-1]}_scores = classifier_weights.loc["
            f"[('{var_name}', str(v)) for v in {var_name}]]"
        )


def model_to_code(
    model,
    idx_w,
    idx_t,
    embed_csv=False,
    embed_enums=False,
    unembed_csv=True,
    one_hot=False,
    autoregressive=False,
    var_types=None,
    output_dir=".",
    name="program",
    example="",
    examples=[],
    do_black=True,
    save=True,
    compress=True,
    unembed_mask=True,
):
    if var_types == True:
        var_types = get_var_types(
            model, idx_w, one_hot=one_hot, enums=embed_enums
        )
    header = []
    if embed_csv or unembed_csv:
        header.append("import numpy as np")
        header.append("import pandas as pd")

    if embed_enums:
        enum_fn = Path(output_dir) / (
            f"{name}_embeddings" if name else "embeddings"
        )
        s = str(enum_fn).replace("/", ".")
        header.append(
            f"from {s} import "
            + ", ".join([f"Emb{i}" for i in range(model.embed.n_vars)])
        )
        # header += ["\n"] + get_embed_enum(model.embed.n_vars)

    header += ["\n"] + get_select_closest(autoregressive=autoregressive)
    if model.blocks[0].n_heads_num:
        header += ["\n"] + get_select(autoregressive=autoregressive)
    header += ["\n"] + get_aggregate()
    if model.blocks[0].n_heads_num:
        header += ["\n"] + get_aggregate_sum()

    unembed_fn = "weights.csv"
    if unembed_csv:
        unembed_df = get_unembed_df(
            model,
            idx_t,
            var_types=var_types,
            one_hot=one_hot,
            enums=embed_enums,
            unembed_mask=unembed_mask,
        )
        unembed_fn = fn = Path(output_dir) / (
            f"{name}_weights.csv" if name else "classifier_weights.csv"
        )
        if save:
            print(f"writing {unembed_df.shape} classifier weights to {fn}")
            unembed_df.to_csv(fn)

    x_name = "tokens" if var_types is not None else "x"
    if model.pool_outputs:
        header += ["\n", f"def run({x_name}, return_details=False):"]
    else:
        header += ["\n", f"def run({x_name}):"]

    lines = [
        "\n",
        (f"# classifier weights " + "#" * 42),
        (
            f"classifier_weights = pd.read_csv('{unembed_fn}', "
            "index_col=[0, 1], dtype={'feature': str})"
        ),
    ]

    lines.append(f"# inputs " + "#" * 53)
    if one_hot:
        lines.append(get_score("tokens"))
    lines += ["\n", f"positions = list(range(len({x_name})))"]
    lines.append(get_score("positions"))
    lines += ["\n", f"ones = [1 for _ in range(len({x_name}))]"]
    lines.append(get_score("ones"))

    if (not one_hot) and (embed_csv or embed_enums):
        embed_df = get_embed_df(model.embed, idx_w)
        embed_fn = fn = Path(output_dir) / (
            f"{name}_embeddings.csv" if name else "embeddings.csv"
        )
        if save:
            print(f"writing {embed_df.shape} embeddings weights to {fn}")
            embed_df.to_csv(fn)
        lines += ["# embed " + "#" * 54]
        if embed_csv:
            lines += embed_csv_to_code(model.embed, idx_w, fn=fn).split("\n")
            for i in range(model.n_vars_cat):
                lines.append(get_score(f"var{i}_embeddings"))
        else:
            enum_fn = fn = Path(output_dir) / (
                f"{name}_embeddings.py" if name else "embeddings.py"
            )
            embed_enum_file = embed_enum_to_file(model.embed, idx_w)
            print(f"writing embedding enums to {fn}")
            with open(enum_fn, "w") as f:
                f.write(embed_enum_file)
            lines += embed_enum_to_code(model.embed, fn=embed_fn).split("\n")
            for i in range(model.n_vars_cat):
                lines.append(get_score(f"var{i}_embeddings"))
    elif var_types is None:
        lines += ["# embed " + "#" * 54]
        lines += embed_to_code(
            model.embed, idx_w, one_hot=one_hot, var_types=var_types
        ).split("\n")
    n_heads = model.blocks[0].cat_attn.W_K.n_heads
    n_layers = len(model.blocks)
    for l in range(n_layers):
        for h in range(model.blocks[0].n_heads_cat):
            lines += [f"# attn_{l}_{h} " + "#" * 52]
            lines += cat_head_to_code(
                model,
                layer=l,
                head=h,
                autoregressive=autoregressive,
                idx_w=idx_w if one_hot else None,
                var_types=var_types,
                one_hot=one_hot,
                compress=compress,
            ).split("\n")
            lines.append(get_score(f"attn_{l}_{h}_outputs"))
            lines.append("\n")
        for h in range(model.blocks[0].n_heads_num):
            lines += [f"# num_attn_{l}_{h} " + "#" * 52]
            lines += num_head_to_code(
                model,
                layer=l,
                head=h,
                autoregressive=autoregressive,
                idx_w=idx_w if one_hot else None,
                var_types=var_types,
                one_hot=one_hot,
                compress=compress,
            ).split("\n")
            lines.append(get_score(f"num_attn_{l}_{h}_outputs"))
            lines.append("\n")
        block = model.blocks[l]
        for h in range(block.n_cat_mlps):
            lines += cat_mlp_to_code(
                model,
                layer=l,
                n_mlp=h,
                var_types=var_types,
                compress=compress,
                one_hot=one_hot,
            ).split("\n")
            lines.append(get_score(f"mlp_{l}_{h}_outputs"))
            lines.append("\n")
        for h in range(block.n_num_mlps):
            lines += num_mlp_to_code(
                model,
                layer=l,
                n_mlp=h,
                var_types=var_types,
                compress=compress,
                one_hot=one_hot,
            ).split("\n")
            lines.append(get_score(f"num_mlp_{l}_{h}_outputs"))
            lines.append("\n")

    _, _, var_names = get_var_names(model, one_hot=one_hot)
    var_dim = model.d_var

    score_names = [f"{var_name[:-1]}_scores" for var_name in var_names]
    lines += [
        (
            f"feature_logits = pd.concat([df.reset_index() for df in "
            f"[{', '.join(score_names)}]])"
        )
    ]
    lines += [
        f"logits = feature_logits.groupby(level=0).sum(numeric_only=True).to_numpy()"
    ]
    if model.pool_outputs:
        lines += [
            f"classes = classifier_weights.columns.to_numpy()",
            f"token_predictions = classes[logits.argmax(-1)]",
            f"pooled = logits.mean(0)",
            f"prediction = classes[pooled.argmax(-1)]",
            f"if return_details:",
            f"\treturn prediction, logits, feature_logits, token_predictions",
            f"return prediction",
        ]
    else:
        lines += [
            f"classes = classifier_weights.columns.to_numpy()",
            f"predictions = classes[logits.argmax(-1)]",
            f"if tokens[0] == '<s>':",
            f"\tpredictions[0] = '<s>'",
            f"if tokens[-1] == '</s>':",
            f"\tpredictions[-1] = '</s>'",
            f"return predictions.tolist()",
        ]
    s = "\n".join(header + ["\t" + s for s in lines])

    if example:
        s += f"\nprint(run({example}))"

    if examples:
        ex_lines = [
            "\n",
            f"examples = {examples}",
            "for x, y in examples:",
            "\tprint(f'x: {x}')",
            "\tprint(f'y: {y}')",
            "\ty_hat = run(x)",
            "\tprint(f'y_hat: {y_hat}')",
            "\tprint()",
        ]
        s += "\n".join(ex_lines)

    m = s.replace("\t", "    ")

    if embed_enums:
        m = re.sub(r'[\'"](Emb[^\'"]+)[\'"]', r"\1", m)

    if do_black:
        m = format_str(m, mode=FileMode())

    fn = Path(output_dir) / f"{name}.py"
    length = m.count("\n")
    if save:
        print(f"writing {length} lines to {fn}")
        with open(fn, "w") as f:
            f.write(m)
    return m
