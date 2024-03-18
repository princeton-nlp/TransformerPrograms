import numpy as np
import pandas as pd
from programs.trec.trec_embeddings import Emb0, Emb1, Emb2, Emb3


def select_closest(keys, queries, predicate):
    scores = [[False for _ in keys] for _ in queries]
    for i, q in enumerate(queries):
        matches = [j for j, k in enumerate(keys) if predicate(q, k)]
        if not (any(matches)):
            scores[i][0] = True
        else:
            j = min(matches, key=lambda j: len(matches) if j == i else abs(i - j))
            scores[i][j] = True
    return scores


def aggregate(attention, values):
    return [[v for a, v in zip(attn, values) if a][0] for attn in attention]


def run(tokens, return_details=False):
    # classifier weights ##########################################
    classifier_weights = pd.read_csv(
        "programs/trec/trec_weights.csv", index_col=[0, 1], dtype={"feature": str}
    )
    # inputs #####################################################

    positions = list(range(len(tokens)))
    position_scores = classifier_weights.loc[[("positions", str(v)) for v in positions]]

    ones = [1 for _ in range(len(tokens))]
    one_scores = classifier_weights.loc[[("ones", "_") for v in ones]].mul(ones, axis=0)
    # embed ######################################################
    embed_df = pd.read_csv("programs/trec/trec_embeddings.csv").set_index("word")
    embeddings = embed_df.loc[tokens]
    var0_embeddings = [Emb0[f"V{i:02}"] for i in embeddings["var0_embeddings"]]
    var1_embeddings = [Emb1[f"V{i:02}"] for i in embeddings["var1_embeddings"]]
    var2_embeddings = [Emb2[f"V{i:02}"] for i in embeddings["var2_embeddings"]]
    var3_embeddings = [Emb3[f"V{i:02}"] for i in embeddings["var3_embeddings"]]
    var0_embedding_scores = classifier_weights.loc[
        [("var0_embeddings", str(v)) for v in var0_embeddings]
    ]
    var1_embedding_scores = classifier_weights.loc[
        [("var1_embeddings", str(v)) for v in var1_embeddings]
    ]
    var2_embedding_scores = classifier_weights.loc[
        [("var2_embeddings", str(v)) for v in var2_embeddings]
    ]
    var3_embedding_scores = classifier_weights.loc[
        [("var3_embeddings", str(v)) for v in var3_embeddings]
    ]

    # attn_0_0 ####################################################
    def predicate_0_0(var1_embedding, position):
        if var1_embedding in {Emb1.V16, Emb1.V20, Emb1.V13, Emb1.V00, Emb1.V22}:
            return position == 2
        elif var1_embedding in {
            Emb1.V23,
            Emb1.V07,
            Emb1.V03,
            Emb1.V02,
            Emb1.V05,
            Emb1.V01,
            Emb1.V12,
            Emb1.V11,
            Emb1.V24,
            Emb1.V15,
        }:
            return position == 1
        elif var1_embedding in {
            Emb1.V06,
            Emb1.V04,
            Emb1.V18,
            Emb1.V14,
            Emb1.V09,
            Emb1.V08,
            Emb1.V10,
        }:
            return position == 4
        elif var1_embedding in {Emb1.V17}:
            return position == 6
        elif var1_embedding in {Emb1.V19}:
            return position == 5
        elif var1_embedding in {Emb1.V21}:
            return position == 18

    attn_0_0_pattern = select_closest(positions, var1_embeddings, predicate_0_0)
    attn_0_0_outputs = aggregate(attn_0_0_pattern, var0_embeddings)
    attn_0_0_output_scores = classifier_weights.loc[
        [("attn_0_0_outputs", str(v)) for v in attn_0_0_outputs]
    ]

    # attn_0_1 ####################################################
    def predicate_0_1(var2_embedding, position):
        if var2_embedding in {
            Emb2.V16,
            Emb2.V13,
            Emb2.V22,
            Emb2.V15,
            Emb2.V23,
            Emb2.V06,
            Emb2.V02,
            Emb2.V00,
            Emb2.V24,
            Emb2.V20,
        }:
            return position == 2
        elif var2_embedding in {
            Emb2.V04,
            Emb2.V12,
            Emb2.V01,
            Emb2.V03,
            Emb2.V07,
            Emb2.V21,
            Emb2.V17,
        }:
            return position == 1
        elif var2_embedding in {Emb2.V05}:
            return position == 21
        elif var2_embedding in {Emb2.V08, Emb2.V11}:
            return position == 5
        elif var2_embedding in {Emb2.V09, Emb2.V10}:
            return position == 3
        elif var2_embedding in {Emb2.V19, Emb2.V14}:
            return position == 4
        elif var2_embedding in {Emb2.V18}:
            return position == 17

    attn_0_1_pattern = select_closest(positions, var2_embeddings, predicate_0_1)
    attn_0_1_outputs = aggregate(attn_0_1_pattern, var0_embeddings)
    attn_0_1_output_scores = classifier_weights.loc[
        [("attn_0_1_outputs", str(v)) for v in attn_0_1_outputs]
    ]

    # attn_0_2 ####################################################
    def predicate_0_2(q_position, k_position):
        if q_position in {0, 1, 2, 4, 5, 8, 19, 22}:
            return k_position == 4
        elif q_position in {3, 20, 6}:
            return k_position == 1
        elif q_position in {7, 9, 10, 13, 15, 18, 21, 23, 24}:
            return k_position == 2
        elif q_position in {11}:
            return k_position == 8
        elif q_position in {12, 14}:
            return k_position == 5
        elif q_position in {16}:
            return k_position == 3
        elif q_position in {17}:
            return k_position == 20

    attn_0_2_pattern = select_closest(positions, positions, predicate_0_2)
    attn_0_2_outputs = aggregate(attn_0_2_pattern, var2_embeddings)
    attn_0_2_output_scores = classifier_weights.loc[
        [("attn_0_2_outputs", str(v)) for v in attn_0_2_outputs]
    ]

    # attn_0_3 ####################################################
    def predicate_0_3(q_position, k_position):
        if q_position in {0, 3, 5, 17, 21, 22, 24}:
            return k_position == 1
        elif q_position in {1, 2, 4, 6, 11, 13, 14, 18, 20}:
            return k_position == 2
        elif q_position in {16, 9, 7}:
            return k_position == 5
        elif q_position in {8, 12, 15, 19, 23}:
            return k_position == 4
        elif q_position in {10}:
            return k_position == 7

    attn_0_3_pattern = select_closest(positions, positions, predicate_0_3)
    attn_0_3_outputs = aggregate(attn_0_3_pattern, var1_embeddings)
    attn_0_3_output_scores = classifier_weights.loc[
        [("attn_0_3_outputs", str(v)) for v in attn_0_3_outputs]
    ]

    # mlp_0_0 #####################################################
    def mlp_0_0(attn_0_0_output, attn_0_3_output):
        key = (attn_0_0_output, attn_0_3_output)
        if key in {
            (Emb0.V00, Emb1.V22),
            (Emb0.V02, Emb1.V14),
            (Emb0.V02, Emb1.V16),
            (Emb0.V02, Emb1.V22),
            (Emb0.V03, Emb1.V00),
            (Emb0.V03, Emb1.V02),
            (Emb0.V03, Emb1.V03),
            (Emb0.V03, Emb1.V04),
            (Emb0.V03, Emb1.V05),
            (Emb0.V03, Emb1.V07),
            (Emb0.V03, Emb1.V08),
            (Emb0.V03, Emb1.V10),
            (Emb0.V03, Emb1.V11),
            (Emb0.V03, Emb1.V13),
            (Emb0.V03, Emb1.V14),
            (Emb0.V03, Emb1.V15),
            (Emb0.V03, Emb1.V16),
            (Emb0.V03, Emb1.V17),
            (Emb0.V03, Emb1.V18),
            (Emb0.V03, Emb1.V22),
            (Emb0.V03, Emb1.V24),
            (Emb0.V04, Emb1.V00),
            (Emb0.V04, Emb1.V02),
            (Emb0.V04, Emb1.V03),
            (Emb0.V04, Emb1.V04),
            (Emb0.V04, Emb1.V05),
            (Emb0.V04, Emb1.V08),
            (Emb0.V04, Emb1.V13),
            (Emb0.V04, Emb1.V14),
            (Emb0.V04, Emb1.V16),
            (Emb0.V04, Emb1.V17),
            (Emb0.V04, Emb1.V18),
            (Emb0.V04, Emb1.V22),
            (Emb0.V04, Emb1.V24),
            (Emb0.V05, Emb1.V00),
            (Emb0.V05, Emb1.V01),
            (Emb0.V05, Emb1.V02),
            (Emb0.V05, Emb1.V03),
            (Emb0.V05, Emb1.V04),
            (Emb0.V05, Emb1.V05),
            (Emb0.V05, Emb1.V07),
            (Emb0.V05, Emb1.V08),
            (Emb0.V05, Emb1.V09),
            (Emb0.V05, Emb1.V10),
            (Emb0.V05, Emb1.V11),
            (Emb0.V05, Emb1.V12),
            (Emb0.V05, Emb1.V13),
            (Emb0.V05, Emb1.V14),
            (Emb0.V05, Emb1.V15),
            (Emb0.V05, Emb1.V16),
            (Emb0.V05, Emb1.V17),
            (Emb0.V05, Emb1.V18),
            (Emb0.V05, Emb1.V20),
            (Emb0.V05, Emb1.V21),
            (Emb0.V05, Emb1.V22),
            (Emb0.V05, Emb1.V24),
            (Emb0.V06, Emb1.V16),
            (Emb0.V06, Emb1.V22),
            (Emb0.V07, Emb1.V05),
            (Emb0.V07, Emb1.V14),
            (Emb0.V07, Emb1.V18),
            (Emb0.V07, Emb1.V22),
            (Emb0.V07, Emb1.V24),
            (Emb0.V08, Emb1.V02),
            (Emb0.V08, Emb1.V05),
            (Emb0.V08, Emb1.V14),
            (Emb0.V08, Emb1.V16),
            (Emb0.V08, Emb1.V18),
            (Emb0.V08, Emb1.V22),
            (Emb0.V08, Emb1.V24),
            (Emb0.V09, Emb1.V00),
            (Emb0.V09, Emb1.V02),
            (Emb0.V09, Emb1.V05),
            (Emb0.V09, Emb1.V14),
            (Emb0.V09, Emb1.V18),
            (Emb0.V09, Emb1.V22),
            (Emb0.V09, Emb1.V24),
            (Emb0.V10, Emb1.V16),
            (Emb0.V10, Emb1.V22),
            (Emb0.V11, Emb1.V02),
            (Emb0.V11, Emb1.V04),
            (Emb0.V11, Emb1.V14),
            (Emb0.V11, Emb1.V16),
            (Emb0.V11, Emb1.V22),
            (Emb0.V11, Emb1.V24),
            (Emb0.V12, Emb1.V00),
            (Emb0.V12, Emb1.V01),
            (Emb0.V12, Emb1.V02),
            (Emb0.V12, Emb1.V03),
            (Emb0.V12, Emb1.V04),
            (Emb0.V12, Emb1.V05),
            (Emb0.V12, Emb1.V07),
            (Emb0.V12, Emb1.V08),
            (Emb0.V12, Emb1.V09),
            (Emb0.V12, Emb1.V10),
            (Emb0.V12, Emb1.V11),
            (Emb0.V12, Emb1.V12),
            (Emb0.V12, Emb1.V13),
            (Emb0.V12, Emb1.V14),
            (Emb0.V12, Emb1.V15),
            (Emb0.V12, Emb1.V16),
            (Emb0.V12, Emb1.V17),
            (Emb0.V12, Emb1.V18),
            (Emb0.V12, Emb1.V20),
            (Emb0.V12, Emb1.V21),
            (Emb0.V12, Emb1.V22),
            (Emb0.V12, Emb1.V24),
            (Emb0.V13, Emb1.V02),
            (Emb0.V13, Emb1.V05),
            (Emb0.V13, Emb1.V14),
            (Emb0.V13, Emb1.V17),
            (Emb0.V13, Emb1.V18),
            (Emb0.V13, Emb1.V22),
            (Emb0.V13, Emb1.V24),
            (Emb0.V14, Emb1.V05),
            (Emb0.V14, Emb1.V14),
            (Emb0.V14, Emb1.V18),
            (Emb0.V14, Emb1.V22),
            (Emb0.V14, Emb1.V24),
            (Emb0.V15, Emb1.V02),
            (Emb0.V15, Emb1.V14),
            (Emb0.V15, Emb1.V22),
            (Emb0.V16, Emb1.V22),
            (Emb0.V17, Emb1.V02),
            (Emb0.V17, Emb1.V05),
            (Emb0.V17, Emb1.V14),
            (Emb0.V17, Emb1.V18),
            (Emb0.V17, Emb1.V22),
            (Emb0.V17, Emb1.V24),
            (Emb0.V18, Emb1.V22),
            (Emb0.V19, Emb1.V05),
            (Emb0.V19, Emb1.V14),
            (Emb0.V19, Emb1.V18),
            (Emb0.V19, Emb1.V22),
            (Emb0.V19, Emb1.V24),
            (Emb0.V20, Emb1.V02),
            (Emb0.V20, Emb1.V05),
            (Emb0.V20, Emb1.V14),
            (Emb0.V20, Emb1.V16),
            (Emb0.V20, Emb1.V17),
            (Emb0.V20, Emb1.V18),
            (Emb0.V20, Emb1.V22),
            (Emb0.V20, Emb1.V24),
            (Emb0.V21, Emb1.V22),
            (Emb0.V22, Emb1.V02),
            (Emb0.V22, Emb1.V05),
            (Emb0.V22, Emb1.V13),
            (Emb0.V22, Emb1.V14),
            (Emb0.V22, Emb1.V17),
            (Emb0.V22, Emb1.V18),
            (Emb0.V22, Emb1.V22),
            (Emb0.V22, Emb1.V24),
            (Emb0.V23, Emb1.V02),
            (Emb0.V23, Emb1.V14),
            (Emb0.V23, Emb1.V18),
            (Emb0.V23, Emb1.V22),
            (Emb0.V23, Emb1.V24),
            (Emb0.V24, Emb1.V14),
            (Emb0.V24, Emb1.V18),
            (Emb0.V24, Emb1.V22),
            (Emb0.V24, Emb1.V24),
        }:
            return 6
        elif key in {
            (Emb0.V00, Emb1.V00),
            (Emb0.V00, Emb1.V01),
            (Emb0.V00, Emb1.V03),
            (Emb0.V00, Emb1.V04),
            (Emb0.V00, Emb1.V06),
            (Emb0.V00, Emb1.V07),
            (Emb0.V00, Emb1.V09),
            (Emb0.V00, Emb1.V10),
            (Emb0.V00, Emb1.V11),
            (Emb0.V00, Emb1.V13),
            (Emb0.V00, Emb1.V17),
            (Emb0.V00, Emb1.V18),
            (Emb0.V00, Emb1.V19),
            (Emb0.V00, Emb1.V20),
            (Emb0.V00, Emb1.V23),
            (Emb0.V00, Emb1.V24),
            (Emb0.V01, Emb1.V00),
            (Emb0.V01, Emb1.V01),
            (Emb0.V01, Emb1.V02),
            (Emb0.V01, Emb1.V03),
            (Emb0.V01, Emb1.V04),
            (Emb0.V01, Emb1.V06),
            (Emb0.V01, Emb1.V07),
            (Emb0.V01, Emb1.V09),
            (Emb0.V01, Emb1.V10),
            (Emb0.V01, Emb1.V11),
            (Emb0.V01, Emb1.V13),
            (Emb0.V01, Emb1.V16),
            (Emb0.V01, Emb1.V17),
            (Emb0.V01, Emb1.V18),
            (Emb0.V01, Emb1.V19),
            (Emb0.V01, Emb1.V20),
            (Emb0.V01, Emb1.V22),
            (Emb0.V01, Emb1.V23),
            (Emb0.V01, Emb1.V24),
            (Emb0.V02, Emb1.V06),
            (Emb0.V02, Emb1.V19),
            (Emb0.V02, Emb1.V23),
            (Emb0.V03, Emb1.V06),
            (Emb0.V03, Emb1.V09),
            (Emb0.V03, Emb1.V19),
            (Emb0.V03, Emb1.V23),
            (Emb0.V04, Emb1.V06),
            (Emb0.V04, Emb1.V09),
            (Emb0.V04, Emb1.V19),
            (Emb0.V04, Emb1.V23),
            (Emb0.V05, Emb1.V06),
            (Emb0.V05, Emb1.V19),
            (Emb0.V05, Emb1.V23),
            (Emb0.V06, Emb1.V06),
            (Emb0.V06, Emb1.V19),
            (Emb0.V06, Emb1.V23),
            (Emb0.V07, Emb1.V03),
            (Emb0.V07, Emb1.V04),
            (Emb0.V07, Emb1.V06),
            (Emb0.V07, Emb1.V09),
            (Emb0.V07, Emb1.V16),
            (Emb0.V07, Emb1.V19),
            (Emb0.V07, Emb1.V20),
            (Emb0.V07, Emb1.V23),
            (Emb0.V09, Emb1.V06),
            (Emb0.V09, Emb1.V16),
            (Emb0.V09, Emb1.V19),
            (Emb0.V09, Emb1.V23),
            (Emb0.V11, Emb1.V06),
            (Emb0.V11, Emb1.V19),
            (Emb0.V11, Emb1.V23),
            (Emb0.V12, Emb1.V06),
            (Emb0.V12, Emb1.V19),
            (Emb0.V13, Emb1.V06),
            (Emb0.V13, Emb1.V09),
            (Emb0.V13, Emb1.V16),
            (Emb0.V13, Emb1.V19),
            (Emb0.V13, Emb1.V23),
            (Emb0.V14, Emb1.V06),
            (Emb0.V14, Emb1.V09),
            (Emb0.V14, Emb1.V16),
            (Emb0.V14, Emb1.V19),
            (Emb0.V14, Emb1.V20),
            (Emb0.V14, Emb1.V23),
            (Emb0.V15, Emb1.V06),
            (Emb0.V15, Emb1.V09),
            (Emb0.V15, Emb1.V19),
            (Emb0.V15, Emb1.V23),
            (Emb0.V16, Emb1.V04),
            (Emb0.V16, Emb1.V06),
            (Emb0.V16, Emb1.V09),
            (Emb0.V16, Emb1.V16),
            (Emb0.V16, Emb1.V17),
            (Emb0.V16, Emb1.V19),
            (Emb0.V16, Emb1.V20),
            (Emb0.V16, Emb1.V23),
            (Emb0.V17, Emb1.V06),
            (Emb0.V17, Emb1.V09),
            (Emb0.V17, Emb1.V19),
            (Emb0.V17, Emb1.V23),
            (Emb0.V18, Emb1.V06),
            (Emb0.V18, Emb1.V09),
            (Emb0.V18, Emb1.V16),
            (Emb0.V18, Emb1.V19),
            (Emb0.V18, Emb1.V20),
            (Emb0.V18, Emb1.V23),
            (Emb0.V19, Emb1.V06),
            (Emb0.V19, Emb1.V09),
            (Emb0.V19, Emb1.V16),
            (Emb0.V19, Emb1.V19),
            (Emb0.V19, Emb1.V23),
            (Emb0.V20, Emb1.V06),
            (Emb0.V20, Emb1.V19),
            (Emb0.V20, Emb1.V23),
            (Emb0.V21, Emb1.V00),
            (Emb0.V21, Emb1.V03),
            (Emb0.V21, Emb1.V04),
            (Emb0.V21, Emb1.V06),
            (Emb0.V21, Emb1.V09),
            (Emb0.V21, Emb1.V11),
            (Emb0.V21, Emb1.V17),
            (Emb0.V21, Emb1.V19),
            (Emb0.V21, Emb1.V20),
            (Emb0.V21, Emb1.V23),
            (Emb0.V22, Emb1.V06),
            (Emb0.V22, Emb1.V09),
            (Emb0.V22, Emb1.V16),
            (Emb0.V22, Emb1.V19),
            (Emb0.V22, Emb1.V20),
            (Emb0.V22, Emb1.V23),
            (Emb0.V23, Emb1.V06),
            (Emb0.V23, Emb1.V09),
            (Emb0.V23, Emb1.V19),
            (Emb0.V23, Emb1.V23),
            (Emb0.V24, Emb1.V06),
            (Emb0.V24, Emb1.V09),
            (Emb0.V24, Emb1.V19),
            (Emb0.V24, Emb1.V23),
        }:
            return 9
        elif key in {
            (Emb0.V00, Emb1.V02),
            (Emb0.V00, Emb1.V16),
            (Emb0.V12, Emb1.V23),
            (Emb0.V15, Emb1.V16),
            (Emb0.V17, Emb1.V16),
            (Emb0.V21, Emb1.V16),
            (Emb0.V23, Emb1.V16),
        }:
            return 13
        elif key in {
            (Emb0.V06, Emb1.V09),
            (Emb0.V09, Emb1.V09),
            (Emb0.V11, Emb1.V09),
            (Emb0.V24, Emb1.V16),
        }:
            return 14
        elif key in {(Emb0.V06, Emb1.V00), (Emb0.V06, Emb1.V05)}:
            return 17
        elif key in {(Emb0.V02, Emb1.V09), (Emb0.V10, Emb1.V09)}:
            return 20
        return 8

    mlp_0_0_outputs = [
        mlp_0_0(k0, k1) for k0, k1 in zip(attn_0_0_outputs, attn_0_3_outputs)
    ]
    mlp_0_0_output_scores = classifier_weights.loc[
        [("mlp_0_0_outputs", str(v)) for v in mlp_0_0_outputs]
    ]

    # attn_1_0 ####################################################
    def predicate_1_0(var0_embedding, position):
        if var0_embedding in {Emb0.V15, Emb0.V00, Emb0.V02}:
            return position == 13
        elif var0_embedding in {Emb0.V06, Emb0.V01}:
            return position == 7
        elif var0_embedding in {Emb0.V23, Emb0.V03}:
            return position == 5
        elif var0_embedding in {
            Emb0.V12,
            Emb0.V14,
            Emb0.V08,
            Emb0.V16,
            Emb0.V05,
            Emb0.V04,
        }:
            return position == 4
        elif var0_embedding in {Emb0.V11, Emb0.V22, Emb0.V17, Emb0.V07}:
            return position == 2
        elif var0_embedding in {Emb0.V09}:
            return position == 15
        elif var0_embedding in {Emb0.V21, Emb0.V10}:
            return position == 1
        elif var0_embedding in {Emb0.V13}:
            return position == 8
        elif var0_embedding in {Emb0.V18}:
            return position == 9
        elif var0_embedding in {Emb0.V19}:
            return position == 6
        elif var0_embedding in {Emb0.V20}:
            return position == 20
        elif var0_embedding in {Emb0.V24}:
            return position == 3

    attn_1_0_pattern = select_closest(positions, var0_embeddings, predicate_1_0)
    attn_1_0_outputs = aggregate(attn_1_0_pattern, var3_embeddings)
    attn_1_0_output_scores = classifier_weights.loc[
        [("attn_1_0_outputs", str(v)) for v in attn_1_0_outputs]
    ]

    # attn_1_1 ####################################################
    def predicate_1_1(q_position, k_position):
        if q_position in {0, 4, 6, 7, 11, 16}:
            return k_position == 4
        elif q_position in {1, 2, 3, 5, 8, 9, 10, 13, 14, 17, 18, 20, 21, 22, 23, 24}:
            return k_position == 1
        elif q_position in {12}:
            return k_position == 22
        elif q_position in {15}:
            return k_position == 2
        elif q_position in {19}:
            return k_position == 3

    attn_1_1_pattern = select_closest(positions, positions, predicate_1_1)
    attn_1_1_outputs = aggregate(attn_1_1_pattern, var0_embeddings)
    attn_1_1_output_scores = classifier_weights.loc[
        [("attn_1_1_outputs", str(v)) for v in attn_1_1_outputs]
    ]

    # attn_1_2 ####################################################
    def predicate_1_2(attn_0_2_output, position):
        if attn_0_2_output in {
            Emb2.V19,
            Emb2.V14,
            Emb2.V01,
            Emb2.V03,
            Emb2.V15,
            Emb2.V05,
            Emb2.V23,
            Emb2.V00,
            Emb2.V17,
        }:
            return position == 1
        elif attn_0_2_output in {
            Emb2.V08,
            Emb2.V16,
            Emb2.V22,
            Emb2.V07,
            Emb2.V09,
            Emb2.V02,
            Emb2.V18,
            Emb2.V20,
        }:
            return position == 2
        elif attn_0_2_output in {Emb2.V04}:
            return position == 13
        elif attn_0_2_output in {
            Emb2.V12,
            Emb2.V13,
            Emb2.V10,
            Emb2.V06,
            Emb2.V21,
            Emb2.V24,
            Emb2.V11,
        }:
            return position == 4

    attn_1_2_pattern = select_closest(positions, attn_0_2_outputs, predicate_1_2)
    attn_1_2_outputs = aggregate(attn_1_2_pattern, var3_embeddings)
    attn_1_2_output_scores = classifier_weights.loc[
        [("attn_1_2_outputs", str(v)) for v in attn_1_2_outputs]
    ]

    # attn_1_3 ####################################################
    def predicate_1_3(attn_0_0_output, position):
        if attn_0_0_output in {
            Emb0.V23,
            Emb0.V19,
            Emb0.V00,
            Emb0.V22,
            Emb0.V04,
            Emb0.V01,
        }:
            return position == 1
        elif attn_0_0_output in {Emb0.V18, Emb0.V02}:
            return position == 5
        elif attn_0_0_output in {Emb0.V03}:
            return position == 3
        elif attn_0_0_output in {
            Emb0.V07,
            Emb0.V16,
            Emb0.V06,
            Emb0.V05,
            Emb0.V09,
            Emb0.V24,
        }:
            return position == 2
        elif attn_0_0_output in {Emb0.V08}:
            return position == 8
        elif attn_0_0_output in {Emb0.V10}:
            return position == 22
        elif attn_0_0_output in {Emb0.V11}:
            return position == 20
        elif attn_0_0_output in {Emb0.V12}:
            return position == 19
        elif attn_0_0_output in {Emb0.V13}:
            return position == 21
        elif attn_0_0_output in {Emb0.V14}:
            return position == 6
        elif attn_0_0_output in {Emb0.V20, Emb0.V17, Emb0.V15}:
            return position == 4
        elif attn_0_0_output in {Emb0.V21}:
            return position == 24

    attn_1_3_pattern = select_closest(positions, attn_0_0_outputs, predicate_1_3)
    attn_1_3_outputs = aggregate(attn_1_3_pattern, var0_embeddings)
    attn_1_3_output_scores = classifier_weights.loc[
        [("attn_1_3_outputs", str(v)) for v in attn_1_3_outputs]
    ]

    # mlp_1_0 #####################################################
    def mlp_1_0(var2_embedding, attn_0_3_output):
        key = (var2_embedding, attn_0_3_output)
        if key in {
            (Emb2.V03, Emb1.V00),
            (Emb2.V03, Emb1.V04),
            (Emb2.V03, Emb1.V05),
            (Emb2.V03, Emb1.V06),
            (Emb2.V03, Emb1.V19),
            (Emb2.V03, Emb1.V22),
            (Emb2.V04, Emb1.V00),
            (Emb2.V04, Emb1.V04),
            (Emb2.V04, Emb1.V06),
            (Emb2.V04, Emb1.V07),
            (Emb2.V04, Emb1.V09),
            (Emb2.V04, Emb1.V22),
            (Emb2.V05, Emb1.V05),
            (Emb2.V05, Emb1.V06),
            (Emb2.V05, Emb1.V19),
            (Emb2.V05, Emb1.V22),
            (Emb2.V06, Emb1.V00),
            (Emb2.V06, Emb1.V03),
            (Emb2.V06, Emb1.V04),
            (Emb2.V06, Emb1.V05),
            (Emb2.V06, Emb1.V06),
            (Emb2.V06, Emb1.V07),
            (Emb2.V06, Emb1.V09),
            (Emb2.V06, Emb1.V10),
            (Emb2.V06, Emb1.V11),
            (Emb2.V06, Emb1.V12),
            (Emb2.V06, Emb1.V14),
            (Emb2.V06, Emb1.V17),
            (Emb2.V06, Emb1.V19),
            (Emb2.V06, Emb1.V22),
            (Emb2.V07, Emb1.V05),
            (Emb2.V07, Emb1.V06),
            (Emb2.V12, Emb1.V05),
            (Emb2.V12, Emb1.V06),
            (Emb2.V14, Emb1.V05),
            (Emb2.V14, Emb1.V06),
            (Emb2.V14, Emb1.V07),
            (Emb2.V14, Emb1.V22),
            (Emb2.V17, Emb1.V00),
            (Emb2.V17, Emb1.V04),
            (Emb2.V17, Emb1.V05),
            (Emb2.V17, Emb1.V06),
            (Emb2.V17, Emb1.V07),
            (Emb2.V17, Emb1.V09),
            (Emb2.V17, Emb1.V11),
            (Emb2.V17, Emb1.V19),
            (Emb2.V17, Emb1.V22),
            (Emb2.V17, Emb1.V23),
            (Emb2.V18, Emb1.V00),
            (Emb2.V19, Emb1.V00),
            (Emb2.V19, Emb1.V03),
            (Emb2.V19, Emb1.V04),
            (Emb2.V19, Emb1.V05),
            (Emb2.V19, Emb1.V07),
            (Emb2.V19, Emb1.V08),
            (Emb2.V19, Emb1.V09),
            (Emb2.V19, Emb1.V12),
            (Emb2.V19, Emb1.V14),
            (Emb2.V19, Emb1.V19),
            (Emb2.V19, Emb1.V21),
            (Emb2.V19, Emb1.V22),
            (Emb2.V21, Emb1.V00),
            (Emb2.V21, Emb1.V04),
            (Emb2.V21, Emb1.V05),
            (Emb2.V21, Emb1.V06),
            (Emb2.V21, Emb1.V07),
            (Emb2.V21, Emb1.V09),
            (Emb2.V21, Emb1.V19),
            (Emb2.V21, Emb1.V22),
            (Emb2.V22, Emb1.V06),
        }:
            return 0
        elif key in {
            (Emb2.V04, Emb1.V05),
            (Emb2.V04, Emb1.V11),
            (Emb2.V04, Emb1.V19),
            (Emb2.V04, Emb1.V21),
            (Emb2.V06, Emb1.V23),
        }:
            return 11
        return 4

    mlp_1_0_outputs = [
        mlp_1_0(k0, k1) for k0, k1 in zip(var2_embeddings, attn_0_3_outputs)
    ]
    mlp_1_0_output_scores = classifier_weights.loc[
        [("mlp_1_0_outputs", str(v)) for v in mlp_1_0_outputs]
    ]

    feature_logits = pd.concat(
        [
            df.reset_index()
            for df in [
                var0_embedding_scores,
                var1_embedding_scores,
                var2_embedding_scores,
                var3_embedding_scores,
                position_scores,
                attn_0_0_output_scores,
                attn_0_1_output_scores,
                attn_0_2_output_scores,
                attn_0_3_output_scores,
                mlp_0_0_output_scores,
                attn_1_0_output_scores,
                attn_1_1_output_scores,
                attn_1_2_output_scores,
                attn_1_3_output_scores,
                mlp_1_0_output_scores,
                one_scores,
            ]
        ]
    )
    logits = feature_logits.groupby(level=0).sum(numeric_only=True).to_numpy()
    classes = classifier_weights.columns.to_numpy()
    token_predictions = classes[logits.argmax(-1)]
    pooled = logits.mean(0)
    prediction = classes[pooled.argmax(-1)]
    if return_details:
        return prediction, logits, feature_logits, token_predictions
    return prediction


examples = [
    (
        [
            "<s>",
            "How",
            "many",
            "trees",
            "go",
            "into",
            "paper",
            "making",
            "in",
            "a",
            "year",
            "?",
            "</s>",
        ],
        "NUM",
    ),
    (
        [
            "<s>",
            "What",
            "<unk>",
            "are",
            "held",
            "in",
            "New",
            "York",
            "this",
            "week",
            "?",
            "</s>",
        ],
        "ENTY",
    ),
    (
        [
            "<s>",
            "Where",
            "did",
            "the",
            "sport",
            "of",
            "<unk>",
            "originate",
            "?",
            "</s>",
        ],
        "LOC",
    ),
    (
        [
            "<s>",
            "What",
            "kind",
            "of",
            "people",
            "took",
            "part",
            "in",
            "<unk>",
            "'",
            "<unk>",
            "in",
            "Massachusetts",
            "in",
            "<unk>",
            "?",
            "</s>",
        ],
        "HUM",
    ),
    (["<s>", "How", "is", "<unk>", "made", "?", "</s>"], "DESC"),
    (
        [
            "<s>",
            "What",
            "do",
            "<unk>",
            "Aaron",
            ",",
            "Jimmy",
            "Stewart",
            ",",
            "and",
            "<unk>",
            "<unk>",
            "<unk>",
            "have",
            "in",
            "common",
            "?",
            "</s>",
        ],
        "DESC",
    ),
    (
        [
            "<s>",
            "How",
            "many",
            "of",
            "them",
            "are",
            "in",
            "<unk>",
            "Africa",
            "?",
            "</s>",
        ],
        "NUM",
    ),
    (
        [
            "<s>",
            "What",
            "country",
            "was",
            "the",
            "setting",
            "of",
            "You",
            "<unk>",
            "Live",
            "<unk>",
            "?",
            "</s>",
        ],
        "LOC",
    ),
    (
        [
            "<s>",
            "What",
            "is",
            "the",
            "largest",
            "sculpture",
            "in",
            "the",
            "world",
            "?",
            "</s>",
        ],
        "ENTY",
    ),
    (
        [
            "<s>",
            "What",
            "<unk>",
            "U.S.",
            "state",
            "boasts",
            "the",
            "most",
            "<unk>",
            "?",
            "</s>",
        ],
        "LOC",
    ),
]
for x, y in examples:
    print(f"x: {x}")
    print(f"y: {y}")
    y_hat = run(x)
    print(f"y_hat: {y_hat}")
    print()
