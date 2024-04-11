from collections import Counter
# import copy
# from copy import deepcopy
# import itertools
# import math
# from pathlib import Path
import random
import re
import string

import datasets
# import gensim.downloader
from nltk.tokenize import word_tokenize
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
# from sklearn.random_projection import GaussianRandomProjection
# from sklearn.preprocessing import normalize
# import torch
# from torch import nn
# from torch.nn import functional as F

from utils import logging

BOS = "<s>"
EOS = "</s>"
SEP = "<sep>"
PAD = "<pad>"
UNK = "<unk>"

alphabet = list(string.ascii_lowercase)

logger = logging.get_logger(__name__)

def make_addition(dataset_size, vocab_size=10, min_length=1, max_length=5, seed=0):
    vocab = np.array([str(i) for i in range(vocab_size)]) # 0 to vocab_size-1
    sents, tags = [], [] # input sentence, output string
    np.random.seed(seed)

    for _ in range(dataset_size):

        l1 = np.random.randint(1, max_length//2+1)
        l2 = np.random.randint(max(1, min_length-l1), max_length//2+1)
        sent = np.random.choice(vocab, size=l1+l2+1, replace=True).tolist()
        sent[l1] = "+"
        sent_str = "".join(sent)
        sents.append([BOS] + sent)

        t = [PAD] + str(int(sent_str[:l1]) + int(sent_str[l1+1:])).split()
        t += [BOS] * (len(sents[-1]) - len(t))
        tags.append(t)
    return pd.DataFrame({"sent": sents, "tags": tags})


def make_addition_with_hints(dataset_size, vocab_size=10, min_length=1, max_length=5, seed=0):
    vocab = np.array([str(i) for i in range(vocab_size)])
    sents, tags = [], [] # input sentence, output string
    np.random.seed(seed)

    for _ in range(dataset_size):

        l1 = np.random.randint(1, max_length//2+1)
        l2 = np.random.randint(max(1, min_length-l1), max_length//2+1)
        sent = np.random.choice(vocab, size=l1+l2, replace=True).tolist()
        sent.insert(l1, "+")
        
        hint_index = 0
        sent_hint = []
        for item in sent:
            if item.isdigit():
                sent_hint.append(alphabet[hint_index])
                hint_index = (hint_index + 1) % len(alphabet)
            else:
                hint_index = 0
            sent_hint.append(item)
        
        sents.append([BOS] + sent_hint)

        sent_str = "".join(sent)
        ans = list(str(int(sent_str[:l1]) + int(sent_str[l1+1:])))
        ans_hint = []
        hint_index = 0
        for item in ans:
            ans_hint.append(alphabet[hint_index])
            hint_index = (hint_index + 1) % len(alphabet)
            ans_hint.append(item)
        t = [PAD] + ans_hint
        t += [BOS] * (len(sents[-1]) - len(t))
        tags.append(t)
    return pd.DataFrame({"sent": sents, "tags": tags})


def make_reverse(vocab_size, dataset_size, min_length=2, max_length=16, seed=0):
    vocab = np.array([str(i) for i in range(vocab_size - 3)])
    sents, tags = [], []
    np.random.seed(seed)
    for _ in range(dataset_size):
        l = np.random.randint(min_length, max_length - 1)
        sent = np.random.choice(vocab, size=l, replace=True).tolist()
        sents.append([BOS] + sent + [EOS])
        tags.append([PAD] + sent[::-1] + [PAD])
    return pd.DataFrame({"sent": sents, "tags": tags})


def make_hist(vocab_size, dataset_size, min_length=2, max_length=16, seed=0):
    vocab = np.array([str(i) for i in range(vocab_size - 2)])
    sents, tags = [], []
    np.random.seed(seed)
    for _ in range(dataset_size):
        l = np.random.randint(min_length, max_length)
        sent = np.random.choice(vocab, size=l, replace=True).tolist()
        counts = Counter(sent)
        sents.append([BOS] + sent)
        tags.append([PAD] + [str(counts[c]) for c in sent])
    return pd.DataFrame({"sent": sents, "tags": tags})


def make_double_hist(
    vocab_size, dataset_size, min_length=1, max_length=10, seed=0
):
    vocab = np.array([str(i) for i in range(vocab_size - 2)])
    sents, tags = [], []
    np.random.seed(seed)
    for _ in range(dataset_size):
        l = np.random.randint(min_length, max_length)
        sent = np.random.choice(vocab, size=l, replace=True).tolist()
        counts = Counter(sent)
        double_counts = Counter(counts.values())
        sents.append([BOS] + sent)
        tags.append([PAD] + [str(double_counts[counts[c]]) for c in sent])
    return pd.DataFrame({"sent": sents, "tags": tags})


def make_sort(vocab_size, dataset_size, min_length=4, max_length=16, seed=0):
    vocab = np.array([str(i) for i in range(vocab_size - 3)])
    sents, tags = [], []
    np.random.seed(seed)
    for _ in range(dataset_size):
        l = np.random.randint(min_length, max_length - 1)
        sent = np.random.choice(vocab, size=l, replace=True).tolist()
        sents.append([BOS] + sent + [EOS])
        tags.append([PAD] + sorted(sent) + [PAD])
    return pd.DataFrame({"sent": sents, "tags": tags})


def make_most_freq(
    vocab_size, dataset_size, min_length=2, max_length=16, seed=0
):
    vocab = np.array([str(i) for i in range(vocab_size - 2)])
    sents, tags = [], []
    np.random.seed(seed)
    for _ in range(dataset_size):
        l = np.random.randint(min_length, max_length)
        sent = np.random.choice(vocab, size=l, replace=True).tolist()
        counts = Counter(sent)
        first_idx = {}
        for i, c in enumerate(sent):
            if c not in first_idx:
                first_idx[c] = i
        order = sorted([(-counts[c], first_idx[c]) for c in counts])
        sents.append([BOS] + sent)
        t = [PAD] + [sent[i] for _, i in order]
        t += [BOS] * (len(sents[-1]) - len(t))
        tags.append(t)
    return pd.DataFrame({"sent": sents, "tags": tags})


def sample_dyck(vocab_size=1, max_depth=8, min_depth=1):
    vocab = [("(", ")"), ("{", "}")][:vocab_size]
    s = []
    l = np.random.randint(min_depth, max_depth + 1)
    for _ in range(l):
        l, r = vocab[np.random.randint(0, len(vocab))]
        idx = np.random.randint(0, 2)
        if idx == 0:
            s += [l, r]
        else:
            s = [l] + s + [r]
    return s


def tag_dyck_pft(sent):
    tags = []
    count = {"(": 0, "{": 0}
    left = {"(": 0, "{": 0}
    match = {")": "(", "}": "{"}
    stack = []
    for c in sent:
        if c in left:
            stack.append(c)
            tags.append("P")
            continue
        l = match[c]
        if len(stack) == 0 or stack[-1] != l:
            tags += ["F"] * (len(sent) - len(tags))
            return tags
        stack.pop()
        if len(stack) == 0:
            tags.append("T")
        else:
            tags.append("P")
    return tags


def make_dyck_pft(
    vocab_size, dataset_size, min_length=2, max_length=16, seed=0
):
    sents, tags = [], []
    ls = []
    vocab = [c for cs in [("(", ")"), ("{", "}")][:vocab_size] for c in cs]
    np.random.seed(seed)
    for _ in range(dataset_size):
        if np.random.randint(2):
            sent = sample_dyck(
                vocab_size=vocab_size,
                max_depth=max_length // 2,
                min_depth=min_length // 2,
            )
            if len(sent) < max_length - 1:
                sent = (
                    sent
                    + np.random.choice(
                        vocab, size=max_length - len(sent) - 1
                    ).tolist()
                )
            elif len(sent) > max_length - 1:
                sent = sent[: max_length - 1]
        else:
            sent = np.random.choice(vocab, size=max_length - 1).tolist()
        sents.append([BOS] + sent)
        tags.append([PAD] + tag_dyck_pft(sent))
    return pd.DataFrame({"sent": sents, "tags": tags})


def get_tokenizer(train, vocab_size=None, unk=False):
    counts = Counter(w for ws in train["sent"] for w in ws)
    words = []
    for w in [PAD] + ([UNK] if unk else []):
        words.append(w)
    if vocab_size:
        words += [w for w, _ in counts.most_common() if w not in words][
            :vocab_size
        ]
    else:
        words += sorted(c for c in counts.keys() if c not in words)
    idx_w = np.array(words)
    w_idx = {w: i for i, w in enumerate(idx_w)}
    tags = []
    for t in [PAD] + ([UNK] if unk else []):
        tags.append(t)
    tags += sorted(set(t for ts in train["tags"] for t in ts if t not in tags))
    idx_t = np.array(tags)
    t_idx = {t: i for i, t in enumerate(idx_t)}
    return idx_w, w_idx, idx_t, t_idx


def tokenize(sents, w_idx, max_len=None):
    unk_id = w_idx.get(UNK, 0)
    max_len = max(len(s) for s in sents)
    out = []
    for s in sents:
        t = [w_idx.get(c, unk_id) for c in s]
        if len(t) < max_len:
            t += [w_idx[PAD]] * (max_len - len(t))
        out.append(t)
    return np.stack(out, 0)


def prepare_dataset(
    train,
    test,
    val=None,
    vocab_size=None,
    unk=False,
    sent_key="sent",
    tag_key="tags",
    max_len=None,
):
    idx_w, w_idx, idx_t, t_idx = get_tokenizer(
        train, vocab_size=vocab_size, unk=unk
    )
    X_train = tokenize(train[sent_key], w_idx, max_len=max_len)
    Y_train = tokenize(train[tag_key], t_idx, max_len=max_len)
    X_test = tokenize(test[sent_key], w_idx, max_len=max_len)
    Y_test = tokenize(test[tag_key], t_idx, max_len=max_len)
    if val is None:
        return (idx_w, w_idx, idx_t, t_idx, X_train, Y_train, X_test, Y_test)
    X_val = tokenize(val[sent_key], w_idx, max_len=max_len)
    Y_val = tokenize(val[tag_key], t_idx, max_len=max_len)
    return (
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
    )


def get_unique(df):
    seen = set()
    sent, tags = [], []
    for s, t in zip(df["sent"], df["tags"]):
        h = "".join(s)
        if h not in seen:
            seen.add(h)
            sent.append(s)
            tags.append(t)
    return pd.DataFrame({"sent": sent, "tags": tags})


def get_dataset(
    name,
    vocab_size,
    dataset_size,
    train_min_length=4,
    train_max_length=16,
    test_min_length=4,
    test_max_length=16,
    seed=0,
    unk=False,
    test_size=0.1,
):
    fns = {
        "reverse": make_reverse,
        "hist": make_hist,
        "double_hist": make_double_hist,
        "most_freq": make_most_freq,
        "dyck1": make_dyck_pft,
        "dyck2": make_dyck_pft,
        "sort": make_sort,
        "addition": make_addition,
        "addition_hints": make_addition_with_hints,
    }
    if name not in fns:
        raise NotImplementedError(name)
    
    length_gen_flag = False if train_min_length == test_min_length and train_max_length == test_max_length else True
    print(f"length_gen_flag: {length_gen_flag}")
    
    for n in (2, 3, 4, 5):
        train_df = fns[name](
            vocab_size=vocab_size,
            dataset_size=n * dataset_size,
            min_length=train_min_length,
            max_length=train_max_length,
            seed=seed,
        )
        
        if length_gen_flag:
            test_df = fns[name](
                vocab_size=vocab_size,
                dataset_size=n * dataset_size,
                min_length=test_min_length,
                max_length=test_max_length,
                seed=seed+random.randint(0, 10000),
            )
            
            test_df = get_unique(test_df)
            temp_test_size = int(dataset_size * test_size)
            if len(test_df) >= temp_test_size:
                test_df = test_df.iloc[:temp_test_size]
        
        train_df = get_unique(train_df)
        if len(train_df) >= dataset_size:
            train_df = train_df.iloc[:dataset_size]    
            break
    
    if not length_gen_flag:
        train_df, test_df = train_test_split(train_df, test_size=test_size, random_state=seed)
    train_df, val_df = train_test_split(train_df, test_size=test_size, random_state=seed)
    return train_df, test_df, val_df, *prepare_dataset(train_df, test_df, val=val_df, unk=unk)