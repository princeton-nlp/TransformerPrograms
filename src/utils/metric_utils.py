import numpy as np
from seqeval.metrics import precision_score
from seqeval.metrics import recall_score
from seqeval.metrics import f1_score
from seqeval.scheme import IOB2


def eval_induction(X, Y, Y_pred, y_pad_idx=0):
    rows = []
    for y in Y:
        m = []
        seen = set()
        for t in y:
            if t not in seen:
                m.append(False)
                seen.add(t)
            else:
                m.append(True)
        rows.append(m)
    mask = np.stack(m, 0) & (Y != y_pad_idx)
    return (Y == Y_pred)[mask].mean()


def __f1_score(y_true, y_pred, o_idx):
    precision, recall, f1 = 0, 0, 0
    m_p = y_pred != o_idx
    if m_p.sum() > 0:
        precision = (y_true[m_p] == y_pred[m_p]).mean()
    m_r = y_true != o_idx
    if m_r.sum() > 0:
        recall = (y_true[m_r] == y_pred[m_r]).mean()
    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    return {"precision": precision, "recall": recall, "f1": f1}


def conll_score(y_true, y_pred):
    return {
        "precision": precision_score(
            y_true, y_pred, mode="strict", scheme=IOB2
        ),
        "recall": recall_score(y_true, y_pred, mode="strict", scheme=IOB2),
        "f1": f1_score(y_true, y_pred, mode="strict", scheme=IOB2),
    }
