from collections import Counter
import copy
from copy import deepcopy
import itertools
import math
from pathlib import Path
import random
import re
import string

import datasets
import gensim.downloader
from nltk.tokenize import word_tokenize
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.random_projection import GaussianRandomProjection
from sklearn.preprocessing import normalize
import torch
from torch import nn
from torch.nn import functional as F

from utils import logging

logger = logging.get_logger(__name__)


BOS = "<s>"
EOS = "</s>"
SEP = "<sep>"
PAD = "<pad>"
UNK = "<unk>"


def make_induction(
    vocab_size, dataset_size, min_length=4, max_length=16, seed=0, unique=True
):
    size = (vocab_size - 2) // 2
    letters = np.array(list(string.ascii_lowercase[:size]))
    numbers = np.array([str(i) for i in range(size)])
    sents, tags = [], []
    seen = set()
    tries = 0
    np.random.seed(seed)
    while len(sents) < dataset_size:
        # for _ in range(dataset_size):
        l = np.random.randint(min_length // 2, (max_length // 2) + 1)
        cton = np.random.randint(0, len(numbers), size=len(letters))
        cs = np.random.randint(0, size, size=l)
        s = [BOS]
        t = [PAD]
        for c in cs:
            s += [letters[c], numbers[cton[c]]]
            t += [UNK if letters[c] not in s[:-2] else numbers[cton[c]], PAD]
        if np.random.randint(0, 2):
            s = [BOS, np.random.choice(numbers)] + s[1:]
            t = [PAD] + t
        else:
            c = np.random.randint(0, size)
            s += [letters[c]]
            t += [UNK if letters[c] not in s[:-2] else numbers[cton[c]]]
        h = "".join(s)
        if h not in seen or (not unique):
            sents.append(s)
            tags.append(t)
            seen.add(h)
        else:
            tries += 1
            if tries > dataset_size * 2:
                break
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


def load_conll(fn, cols=["ignore", "word", "ignore", "pos"], sep="\t"):
    with open(fn, "r") as f:
        text = f.read()
    chunks = text.split("\n\n")
    lst = []
    for chunk in chunks:
        lines = [l.strip() for l in chunk.split("\n")]
        sent = {k: [] for k in cols if k != "ignore"}
        for line in lines:
            parts = line.split(sep)
            for c, s in zip(cols, parts):
                if c != "ignore":
                    sent[c].append(s)
        lst.append(sent)
    return lst


def get_conll_ner(
    name,
    vocab_size,
    dataset_size=None,
    min_length=1,
    max_length=64,
    seed=0,
    unk=True,
    do_lower=False,
    replace_numbers=False,
    get_val=True,
):
    if name == "conll_pos":
        data = datasets.load_dataset("conll2003")
        t_idx = {
            '"': 0,
            "''": 1,
            "#": 2,
            "$": 3,
            "(": 4,
            ")": 5,
            ",": 6,
            ".": 7,
            ":": 8,
            "``": 9,
            "CC": 10,
            "CD": 11,
            "DT": 12,
            "EX": 13,
            "FW": 14,
            "IN": 15,
            "JJ": 16,
            "JJR": 17,
            "JJS": 18,
            "LS": 19,
            "MD": 20,
            "NN": 21,
            "NNP": 22,
            "NNPS": 23,
            "NNS": 24,
            "NN|SYM": 25,
            "PDT": 26,
            "POS": 27,
            "PRP": 28,
            "PRP$": 29,
            "RB": 30,
            "RBR": 31,
            "RBS": 32,
            "RP": 33,
            "SYM": 34,
            "TO": 35,
            "UH": 36,
            "VB": 37,
            "VBD": 38,
            "VBG": 39,
            "VBN": 40,
            "VBP": 41,
            "VBZ": 42,
            "WDT": 43,
            "WP": 44,
            "WP$": 45,
            "WRB": 46,
        }
        tag_col = "pos_tags"
    elif name == "conll_chunk":
        data = datasets.load_dataset("conll2000")
        t_idx = {
            "O": 0,
            "B-ADJP": 1,
            "I-ADJP": 2,
            "B-ADVP": 3,
            "I-ADVP": 4,
            "B-CONJP": 5,
            "I-CONJP": 6,
            "B-INTJ": 7,
            "I-INTJ": 8,
            "B-LST": 9,
            "I-LST": 10,
            "B-NP": 11,
            "I-NP": 12,
            "B-PP": 13,
            "I-PP": 14,
            "B-PRT": 15,
            "I-PRT": 16,
            "B-SBAR": 17,
            "I-SBAR": 18,
            "B-UCP": 19,
            "I-UCP": 20,
            "B-VP": 21,
            "I-VP": 22,
        }
        tag_col = "chunk_tags"
    else:
        data = datasets.load_dataset("conll2003")
        t_idx = {
            "O": 0,
            "B-PER": 1,
            "I-PER": 2,
            "B-ORG": 3,
            "I-ORG": 4,
            "B-LOC": 5,
            "I-LOC": 6,
            "B-MISC": 7,
            "I-MISC": 8,
        }
        tag_col = "ner_tags"

    idx_t = {v: k for k, v in t_idx.items()}
    idx_t = np.array([idx_t[i] for i in range(len(idx_t))])
    if "validation" in data:
        train, test, val = (
            data["train"].to_pandas(),
            data["test"].to_pandas(),
            data["validation"].to_pandas(),
        )
    else:
        train_full, test = data["train"].to_pandas(), data["test"].to_pandas()
        np.random.seed(0)
        train, val = train_test_split(train_full, test_size=0.1, random_state=0)
        logger.info(
            f"no validation set for {name}, splitting "
            f"{len(train_full)} -> {len(train)}/{len(val)}"
        )
    f = lambda lst: len(lst) >= min_length and len(lst) < max_length - 1
    lst = []

    def fmt(w):
        s = w
        if do_lower:
            s = s.lower()
        if replace_numbers:
            s = re.sub(r"[0-9]+", "@", s)
        return s

    for d in (train, test, val):
        sents = [
            [BOS] + [fmt(w) for w in wds] + [EOS]
            for wds in d["tokens"]
            if f(wds)
        ]
        tags = [
            [PAD] + idx_t[ts].tolist() + [PAD] for ts in d[tag_col] if f(ts)
        ]
        lst.append(pd.DataFrame({"sent": sents, "tags": tags}))
    logger.info(f"kept {len(lst[0])}/{len(train)} training examples")
    train, test, val = lst
    if (dataset_size or 0) > 0 and dataset_size < len(train):
        random.seed(seed)
        train = random.sample(train, dataset_size)
    if get_val:
        return (
            train,
            test,
            val,
            *prepare_dataset(
                train,
                test,
                val=val,
                vocab_size=vocab_size,
                unk=unk,
            ),
        )
    return (
        train,
        test,
        *prepare_dataset(train, test, vocab_size=vocab_size, unk=unk),
    )


def get_classification_dataset(
    name,
    vocab_size,
    dataset_size=None,
    min_length=4,
    max_length=32,
    seed=0,
    unk=True,
    do_lower=False,
    get_val=True,
):
    if name == "sst2":
        data = datasets.load_dataset("glue", name)
    elif name == "subj":
        data = datasets.load_dataset("SetFit/subj")
    else:
        data = datasets.load_dataset(name)
    if "validation" in data:
        train, val, test = (
            data["train"].to_pandas(),
            data["validation"].to_pandas(),
            data["test"].to_pandas(),
        )
    elif get_val:
        train_full, test = data["train"].to_pandas(), data["test"].to_pandas()
        np.random.seed(0)
        train, val = train_test_split(train_full, test_size=0.1, random_state=0)
        logger.info(
            f"no validation set for {name}, splitting "
            f"{len(train_full)} -> {len(train)}/{len(val)}"
        )
    else:
        train, test = data["train"].to_pandas(), data["test"].to_pandas()
        val = test
    f = lambda lst: len(lst) >= min_length and (
        max_length is None or len(lst) < max_length - 1
    )
    lst = []
    col = "text" if "text" in train.columns else "sentence"
    label_col = "label" if "label" in train.columns else "coarse_label"
    for d in (train, val, test):
        sents, tags = [], []
        for s, _y in zip(d[col], d[label_col]):
            if name == "ag_news":
                y = ["World", "Sports", "Business", "Sci/Tech"][_y]
            elif name == "trec":
                y = ["ABBR", "ENTY", "DESC", "HUM", "LOC", "NUM"][_y]
            elif name == "rotten_tomatoes":
                y = ["Negative", "Positive"][_y]
            elif name == "subj":
                y = ["Objective", "Subjective"][_y]
            else:
                y = str(_y)
            wds = word_tokenize(s)
            if f(wds):
                sents.append(
                    [BOS] + [w.lower() if do_lower else w for w in wds] + [EOS]
                )
                tags.append([y] + [PAD] * (len(sents[-1]) - 1))
        lst.append(pd.DataFrame({"sent": sents, "tags": tags}))
    logger.info(f"{len(lst[0])}/{len(train)} training examples")
    train, val, test = lst
    if (dataset_size or 0) > 0 and dataset_size < len(train):
        random.seed(seed)
        train = random.sample(train, dataset_size)
    if get_val:
        return (
            train,
            test,
            val,
            *prepare_dataset(
                train, test, val=val, vocab_size=vocab_size, unk=unk
            ),
        )
    return (
        train,
        test,
        *prepare_dataset(train, test, vocab_size=vocab_size, unk=unk),
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
    dataset_size=None,
    min_length=4,
    max_length=16,
    seed=0,
    unk=False,
    test_size=0.1,
    do_lower=True,
    get_val=True,
    replace_numbers=False,
    unique=False,
):
    if name.startswith("conll"):
        return get_conll_ner(
            name,
            vocab_size,
            dataset_size=dataset_size,
            min_length=min_length,
            max_length=max_length,
            seed=seed,
            do_lower=do_lower,
            get_val=get_val,
            replace_numbers=replace_numbers,
        )
    if name in ("rotten_tomatoes", "ag_news", "trec", "subj"):
        return get_classification_dataset(
            name,
            vocab_size,
            dataset_size=dataset_size,
            min_length=min_length,
            max_length=max_length,
            seed=seed,
            do_lower=do_lower,
            get_val=get_val,
        )
    fns = {
        "induction": make_induction,
        "reverse": make_reverse,
        "hist": make_hist,
        "double_hist": make_double_hist,
        "most_freq": make_most_freq,
        "dyck1": make_dyck_pft,
        "dyck2": make_dyck_pft,
        "sort": make_sort,
    }
    if name not in fns:
        raise NotImplementedError(name)
    if unique:
        for n in (2, 3, 4, 5):
            df = fns[name](
                vocab_size=vocab_size,
                dataset_size=n * dataset_size,
                min_length=min_length,
                max_length=max_length,
                seed=seed,
            )
            df = get_unique(df)
            if len(df) >= dataset_size:
                df = df.iloc[:dataset_size]
                break
    else:
        df = fns[name](
            vocab_size=vocab_size,
            dataset_size=dataset_size,
            min_length=min_length,
            max_length=max_length,
            seed=seed,
        )
    train, test = train_test_split(df, test_size=test_size, random_state=seed)
    if get_val:
        train, val = train_test_split(
            train, test_size=test_size, random_state=seed
        )
        return train, test, val, *prepare_dataset(train, test, val=val, unk=unk)
    return train, test, *prepare_dataset(train, test, unk=unk)


class LocalGlove:
    def __init__(self, fn, idx_w=None):
        rows = []
        self.key_to_index = {}
        need = set(idx_w) if idx_w is not None else None
        with open(fn, "r") as f:
            for line in f:
                i = line.find(" ")
                w = line[:i]
                if (not need) or w in need:
                    parts = line.strip().split(" ")
                    self.key_to_index[parts[0]] = len(rows)
                    rows.append(np.array([float(v) for v in parts[1:]]))
        self.vectors = np.stack(rows, 0)
        logger.info(f"loaded {len(self.vectors)} rows from {fn}")


def get_glove_embeddings(
    idx_w,
    name="glove-wiki-gigaword-100",
    dim=None,
):
    if name.startswith("data"):
        glove_vectors = LocalGlove(name, idx_w)
    else:
        glove_vectors = gensim.downloader.load(name)
    lst = []
    V = glove_vectors.vectors
    missing = []
    for w_ in idx_w:
        if name.startswith("data"):
            w = w_
        else:
            w = w_.lower()
        if w in glove_vectors.key_to_index:
            lst.append(V[glove_vectors.key_to_index[w]])
        else:
            lst.append(np.random.randn(V.shape[1]))
            missing.append(w)
    logger.info(f"found {len(lst)-len(missing)}/{len(lst)} glove embeddings")
    logger.info(f"missing {missing[:10] + ['...']}")
    emb = np.stack(lst, 0)
    if dim is not None and dim != emb.shape[-1]:
        emb = GaussianRandomProjection(
            n_components=dim, random_state=0
        ).fit_transform(emb)
    return emb
