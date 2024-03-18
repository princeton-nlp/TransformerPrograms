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
import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.models.transformers import Transformer
from src.models.programs import (
    TransformerProgramModel,
    argmax,
    gumbel_hard,
    gumbel_soft,
    softmax,
)
from src.utils import code_utils, data_utils, logging, metric_utils

logger = logging.get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()

    # Output
    parser.add_argument("--output_dir", type=str, default="output/scratch")

    # Data
    parser.add_argument("--dataset", type=str, default="reverse")
    parser.add_argument("--vocab_size", type=int, default=8)
    parser.add_argument("--dataset_size", type=int, default=-1)
    parser.add_argument("--min_length", type=int, default=1)
    parser.add_argument("--max_length", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--do_lower", type=int, default=0)
    parser.add_argument("--unique", type=int, default=1)
    parser.add_argument("--replace_numbers", type=int, default=0)

    # Model
    parser.add_argument("--n_vars_cat", type=int, default=1)
    parser.add_argument("--n_vars_num", type=int, default=1)
    parser.add_argument("--d_var", type=int, default=None)
    parser.add_argument("--n_heads_cat", type=int, default=2)
    parser.add_argument("--n_heads_num", type=int, default=2)
    parser.add_argument("--d_mlp", type=int, default=64)
    parser.add_argument("--n_cat_mlps", type=int, default=1)
    parser.add_argument("--n_num_mlps", type=int, default=1)
    parser.add_argument("--mlp_vars_in", type=int, default=2)
    parser.add_argument("--n_layers", type=int, default=1)
    parser.add_argument("--sample_fn", type=str, default="gumbel_soft")
    parser.add_argument("--one_hot_embed", action="store_true")
    parser.add_argument("--count_only", action="store_true")
    parser.add_argument("--selector_width", type=int, default=0)
    parser.add_argument("--attention_type", type=str, default="cat")
    parser.add_argument("--rel_pos_bias", type=str, default="fixed")
    parser.add_argument("--mlp_type", type=str, default="cat")
    parser.add_argument("--autoregressive", action="store_true")

    parser.add_argument(
        "--glove_embeddings", type=str, default="data/glove.840B.300d.txt"
    )
    parser.add_argument("--do_glove", type=int, default=0)

    parser.add_argument("--unembed_mask", type=int, default=1)
    parser.add_argument("--pool_outputs", type=int, default=0)

    # Standard model
    parser.add_argument("--standard", action="store_true")
    parser.add_argument("--d_model", type=int, default=64)
    parser.add_argument("--d_head", type=int, default=None)
    parser.add_argument("--n_heads", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.0)

    # Training
    parser.add_argument("--lr", type=float, default=5e-2)
    parser.add_argument("--max_grad_norm", type=float, default=None)
    parser.add_argument("--gumbel_samples", type=int, default=1)
    parser.add_argument("--n_epochs", type=int, default=250)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--tau_init", type=float, default=3.0)
    parser.add_argument("--tau_end", type=float, default=0.01)
    parser.add_argument("--tau_schedule", type=str, default="geomspace")
    parser.add_argument("--loss_agg", type=str, default="per_token")

    parser.add_argument("--save", action="store_true")
    parser.add_argument("--save_code", action="store_true")

    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()

    if "dyck1" in args.dataset:
        args.autoregressive = True
        args.vocab_size = 1
    if "dyck2" in args.dataset:
        args.autoregressive = True
        args.vocab_size = 2

    logging.initialize(args.output_dir)

    if args.standard and args.d_head is None:
        args.d_head = int(args.d_model // args.n_heads)
        logger.info(
            f"setting d_head to {args.d_model} // {args.n_heads} = {args.d_head}"
        )

    return args


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def run_training(
    model,
    opt,
    X_train,
    Y_train,
    X_test=None,
    Y_test=None,
    eval_splits=None,
    batch_size=256,
    n_epochs=5,
    temps=None,
    n_samples=1,
    x_pad_idx=0,
    y_pad_idx=0,
    autoregressive=False,
    loss_agg="per_token",
    max_grad_norm=None,
    reg_alpha=None,
    patience=None,
    o_idx=None,
    idx_t=None,
    smooth_temps=True,
):
    train_dataloader = DataLoader(
        list(zip(X_train, Y_train)), batch_size=batch_size, shuffle=True
    )
    out = []
    metrics = []
    t = tqdm(range(n_epochs), total=n_epochs)
    if temps is None:
        temps = [None] * n_epochs
    if eval_splits is None and X_test is not None:
        eval_splits = [("val", X_test, Y_test)]
    for epoch in t:
        temp = temps[epoch]
        model.train()
        if temp is not None and smooth_temps and epoch + 1 < len(temps):
            ttemps = np.geomspace(temp, temps[epoch + 1], len(train_dataloader))
        elif temp is not None:
            ttemps = [temp] * len(train_dataloader)
        else:
            ttemps = [None] * len(train_dataloader)
        epoch_losses = []
        for ttemp, (x, y) in zip(ttemps, train_dataloader):
            if ttemp is not None:
                model.set_temp(ttemp)
            x = x.to(model.device)
            m = (x != x_pad_idx).float()
            mask = (m.unsqueeze(-1) @ m.unsqueeze(-2)).bool()
            if autoregressive:
                mask = torch.tril(mask)
            lst = []
            losses_lst = []
            tgts = y.to(model.device)
            for _ in range(n_samples):
                logits = model(x, mask=mask)
                if loss_agg == "per_seq":
                    log_probs = logits.log_softmax(-1)
                    losses = -log_probs.gather(2, tgts.unsqueeze(-1))
                    losses = losses.masked_fill(
                        (tgts == y_pad_idx).unsqueeze(-1), 0.0
                    ).sum(-1)
                else:
                    log_probs = logits.log_softmax(-1)
                    all_losses = -log_probs.gather(
                        2, tgts.unsqueeze(-1)
                    ).squeeze(-1)
                    masked_losses = all_losses.masked_fill(
                        (tgts == y_pad_idx), 0.0
                    )
                    lengths = (tgts != y_pad_idx).sum(-1)
                    losses = masked_losses.sum(-1) / lengths
                loss = losses.mean()
                if reg_alpha:
                    loss += reg_alpha * model.embed.reg()
                lst.append(loss)
                losses_lst.append(losses.detach().cpu())
            loss = torch.stack(lst, 0).mean(0)
            loss.backward()
            if torch.isnan(loss):
                m = torch.isnan(losses_lst[-1])
                print(losses_lst[-1])
                print(m.nonzero())
                print(log_probs[m])
                print(x[m], tgts[m])
                raise ValueError("loss is nan")
            if max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    max_grad_norm,
                )
            epoch_losses.append(torch.stack(losses_lst, 0).mean(0).numpy())
            opt.step()
            opt.zero_grad()
            model.zero_grad()
        epoch_loss = np.concatenate(epoch_losses, 0).mean()
        d = {"loss": epoch_loss.mean()}
        t.set_postfix(d)
        out.append(epoch_loss.mean())
        model.eval()
        with torch.no_grad():
            d = {
                "epoch": epoch,
                "epoch_loss": epoch_loss.mean(),
                "split": "train",
            }
            d["loss"], d["acc"], m = run_test(
                model,
                X_train,
                Y_train,
                x_pad_idx=x_pad_idx,
                y_pad_idx=y_pad_idx,
                autoregressive=autoregressive,
                loss_agg=loss_agg,
                o_idx=o_idx,
                idx_t=idx_t,
            )
            d.update(m)
            metrics.append(d)
            for split, X, Y in eval_splits:
                d = {
                    "epoch": epoch,
                    "epoch_loss": epoch_loss.mean(),
                    "split": split,
                }
                d["loss"], d["acc"], m = run_test(
                    model,
                    X,
                    Y,
                    batch_size=batch_size,
                    x_pad_idx=x_pad_idx,
                    y_pad_idx=y_pad_idx,
                    autoregressive=autoregressive,
                    loss_agg=loss_agg,
                    o_idx=o_idx,
                    idx_t=idx_t,
                )
                d.update(m)
                metrics.append(d)
        if patience is not None and epoch - np.argmin(out) > patience:
            logger.info(f"no improvement for {patience} epochs, stopping")
            break

    return pd.DataFrame(metrics)


def run_test(
    model,
    X,
    Y,
    batch_size=256,
    return_preds=False,
    x_pad_idx=0,
    y_pad_idx=0,
    autoregressive=False,
    func=torch.argmax,
    loss_agg="per_token",
    o_idx=None,
    idx_t=None,
):
    dataloader = DataLoader(
        list(zip(X, Y)), batch_size=batch_size, shuffle=False
    )
    out = []
    preds = []
    true = []
    model.eval()
    for x, y in dataloader:
        x = x.to(model.device)
        m = (x != x_pad_idx).float()
        mask = (m.unsqueeze(-1) @ m.unsqueeze(-2)).bool()
        if autoregressive:
            mask = torch.tril(mask)
        with torch.no_grad():
            log_probs = model(x, mask=mask).log_softmax(-1)
        tgts = y.to(model.device)
        if loss_agg == "per_seq":
            losses = -log_probs.gather(2, tgts.unsqueeze(-1))
            losses = losses.masked_fill(
                (tgts == y_pad_idx).unsqueeze(-1), 0.0
            ).sum(-1)
        else:
            all_losses = -log_probs.gather(2, tgts.unsqueeze(-1)).squeeze(-1)
            masked_losses = all_losses.masked_fill((tgts == y_pad_idx), 0.0)
            lengths = (tgts != y_pad_idx).sum(-1)
            losses = masked_losses.sum(-1) / lengths
        out.append(losses.detach().cpu().numpy())
        pred = func(log_probs, -1)
        preds.append(pred.detach().cpu().numpy())
        true.append(tgts.detach().cpu().numpy())
    preds = np.concatenate(preds, 0)
    true = np.concatenate(true, 0)
    m = true != y_pad_idx
    acc = (preds == true)[m].mean()
    metrics = {}
    if o_idx is not None:
        y_true = [idx_t[y[y != y_pad_idx]].tolist() for y in true]
        y_pred = [
            idx_t[y_hat[y != y_pad_idx]].tolist()
            for y, y_hat in zip(true, preds)
        ]
        metrics = metric_utils.conll_score(y_true=y_true, y_pred=y_pred)
    loss = np.concatenate(out, 0).mean()
    if return_preds:
        return loss, acc, metrics, preds, true
    return loss, acc, metrics


def get_sample_fn(name):
    d = {
        "softmax": softmax,
        "gumbel_hard": gumbel_hard,
        "gumbel_soft": gumbel_soft,
    }
    if name not in d:
        raise NotImplementedError(name)
    return d[name]


def run_program(
    args,
    train=None,
    test=None,
    idx_w=None,
    w_idx=None,
    idx_t=None,
    t_idx=None,
    X_train=None,
    Y_train=None,
    X_test=None,
    Y_test=None,
    X_val=None,
    Y_val=None,
):
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
    ).to(torch.device(args.device))

    opt = Adam([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    n_epochs = args.n_epochs
    if args.tau_schedule not in ("linspace", "geomspace"):
        raise NotImplementedError(args.tau_schedule)
    tau_schedule = (
        np.linspace if args.tau_schedule == "linspace" else np.geomspace
    )
    set_seed(args.seed)
    out = run_training(
        model,
        opt,
        X_train,
        Y_train,
        eval_splits=[("val", X_val, Y_val), ("test", X_test, Y_test)],
        batch_size=args.batch_size,
        n_epochs=n_epochs,
        n_samples=args.gumbel_samples,
        autoregressive=args.autoregressive,
        temps=tau_schedule(args.tau_init, args.tau_end, n_epochs),
        x_pad_idx=w_idx["<pad>"],
        y_pad_idx=t_idx["<pad>"],
        loss_agg=args.loss_agg,
        max_grad_norm=args.max_grad_norm,
        o_idx=t_idx.get("O", None),
        idx_t=idx_t,
    )
    out["sample_fn"] = args.sample_fn
    model.set_temp(args.tau_end, argmax)
    dfs = [out]
    for split, X, Y in [
        ("train", X_train, Y_train),
        ("val", X_val, Y_val),
        ("test", X_test, Y_test),
    ]:
        loss, acc, metrics, preds, true = run_test(
            model,
            X,
            Y,
            return_preds=True,
            x_pad_idx=w_idx["<pad>"],
            y_pad_idx=t_idx["<pad>"],
            autoregressive=args.autoregressive,
            loss_agg=args.loss_agg,
            o_idx=t_idx.get("O", None),
            idx_t=idx_t,
        )
        logger.info(f"{split}: loss={loss}, acc={acc}, metrics={metrics}")
        df = pd.DataFrame(
            {
                "epoch": [n_epochs],
                "split": split,
                "loss": loss,
                "acc": acc,
                "sample_fn": "argmax",
            }
        )
        for k, v in metrics.items():
            df[k] = v
        dfs.append(df)
    df = pd.concat(dfs).reset_index(drop=True)

    if args.save:
        fn = Path(args.output_dir) / "model.pt"
        logger.info(f"saving model to {fn}")
        torch.save(model.state_dict(), str(fn))

    if args.save_code:
        logger.info(f"saving code to {args.output_dir}")
        x = idx_w[X_val[0]]
        x = x[x != "<pad>"].tolist()
        try:
            code_utils.model_to_code(
                model=model,
                idx_w=idx_w,
                idx_t=idx_t,
                embed_csv=not args.one_hot_embed,
                unembed_csv=True,
                one_hot=args.one_hot_embed,
                autoregressive=args.autoregressive,
                var_types=True,
                output_dir=args.output_dir,
                name=args.dataset,
                example=x,
            )
        except Exception as e:
            logger.error(f"error saving code: {e}")

    return df


def run_standard(
    args,
    train=None,
    test=None,
    idx_w=None,
    w_idx=None,
    idx_t=None,
    t_idx=None,
    X_train=None,
    Y_train=None,
    X_test=None,
    Y_test=None,
    X_val=None,
    Y_val=None,
):
    init_emb = None
    if args.glove_embeddings and args.do_glove:
        emb = data_utils.get_glove_embeddings(
            idx_w,
            args.glove_embeddings,
            dim=args.d_model,
        )
        init_emb = torch.tensor(emb, dtype=torch.float32).T
    unembed_mask = None
    if args.unembed_mask:
        unembed_mask = np.array([t in ("<unk>", "<pad>") for t in idx_t])
    model = Transformer(
        d_vocab=len(idx_w),
        d_vocab_out=len(idx_t),
        n_layers=args.n_layers,
        d_model=args.d_model,
        d_mlp=args.d_mlp,
        n_heads=args.n_heads,
        n_ctx=X_train.shape[1],
        dropout=args.dropout,
        init_emb=init_emb,
        unembed_mask=unembed_mask,
        pool_outputs=args.pool_outputs,
    ).to(torch.device(args.device))

    opt = Adam([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    n_epochs = args.n_epochs
    set_seed(args.seed)
    out = run_training(
        model,
        opt,
        X_train,
        Y_train,
        eval_splits=[("val", X_val, Y_val), ("test", X_test, Y_test)],
        batch_size=args.batch_size,
        n_epochs=n_epochs,
        n_samples=1,
        autoregressive=args.autoregressive,
        x_pad_idx=w_idx["<pad>"],
        y_pad_idx=t_idx["<pad>"],
        loss_agg=args.loss_agg,
        o_idx=t_idx.get("O", None),
        idx_t=idx_t,
    )
    dfs = [out]
    for split, X, Y in [
        ("train", X_train, Y_train),
        ("val", X_val, Y_val),
        ("test", X_test, Y_test),
    ]:
        loss, acc, metrics, preds, true = run_test(
            model,
            X,
            Y,
            return_preds=True,
            x_pad_idx=w_idx["<pad>"],
            y_pad_idx=t_idx["<pad>"],
            autoregressive=args.autoregressive,
            loss_agg=args.loss_agg,
            o_idx=t_idx.get("O", None),
            idx_t=idx_t,
        )
        logger.info(f"end ({split}): loss={loss}, acc={acc}, metrics={metrics}")
        df = pd.DataFrame(
            {
                "epoch": [n_epochs],
                "split": split,
                "loss": loss,
                "acc": acc,
            }
        )
        for k, v in metrics.items():
            df[k] = v
        dfs.append(df)
    df = pd.concat(dfs).reset_index(drop=True)

    if args.save:
        fn = Path(args.output_dir) / "model.pt"
        logger.info(f"saving model to {fn}")
        torch.save(model.state_dict(), str(fn))

    return df


def run(args):
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
    logger.info(f"vocab size: {len(idx_w)}")
    logger.info(f"X_train: {X_train.shape}, Y_train, {Y_train.shape}")
    logger.info(f"X_val: {X_val.shape}, Y_val, {Y_val.shape}")
    logger.info(f"X_test: {X_test.shape}, Y_test, {Y_test.shape}")
    a = set(["".join(s) for s in train["sent"]])
    b = set(["".join(s) for s in test["sent"]])
    logger.info(f"{len(a)}/{len(train)} unique training inputs")
    logger.info(f"{len(b - a)}/{len(test)} unique test inputs not in train")

    f = run_standard if args.standard else run_program
    results = f(
        args,
        train=train,
        test=test,
        idx_w=idx_w,
        w_idx=w_idx,
        idx_t=idx_t,
        t_idx=t_idx,
        X_train=X_train,
        Y_train=Y_train,
        X_test=X_test,
        Y_test=Y_test,
        X_val=X_val,
        Y_val=Y_val,
    )
    fn = Path(args.output_dir) / "results.csv"
    logger.info(f"writing results to {fn}")
    results.to_csv(fn, index=False)


if __name__ == "__main__":
    args = parse_args()
    logger.info(f"args: {vars(args)}")
    with open(Path(args.output_dir) / "args.json", "w") as f:
        json.dump(vars(args), f)
    run(args)
