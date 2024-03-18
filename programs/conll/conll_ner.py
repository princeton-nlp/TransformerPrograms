import numpy as np
import pandas as pd
from programs.conll.conll_ner_embeddings import Emb0, Emb1, Emb2, Emb3


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


def run(tokens):
    # classifier weights ##########################################
    classifier_weights = pd.read_csv(
        "programs/conll/conll_ner_weights.csv", index_col=[0, 1], dtype={"feature": str}
    )
    # inputs #####################################################

    positions = list(range(len(tokens)))
    position_scores = classifier_weights.loc[[("positions", str(v)) for v in positions]]

    ones = [1 for _ in range(len(tokens))]
    one_scores = classifier_weights.loc[[("ones", "_") for v in ones]].mul(ones, axis=0)
    # embed ######################################################
    embed_df = pd.read_csv("programs/conll/conll_ner_embeddings.csv").set_index("word")
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
    def predicate_0_0(q_position, k_position):
        if q_position in {0, 1, 3}:
            return k_position == 2
        elif q_position in {2}:
            return k_position == 1
        elif q_position in {4}:
            return k_position == 3
        elif q_position in {5}:
            return k_position == 4
        elif q_position in {6}:
            return k_position == 5
        elif q_position in {7}:
            return k_position == 6
        elif q_position in {8}:
            return k_position == 7
        elif q_position in {9}:
            return k_position == 8
        elif q_position in {10}:
            return k_position == 9
        elif q_position in {11}:
            return k_position == 10
        elif q_position in {12}:
            return k_position == 11
        elif q_position in {13}:
            return k_position == 12
        elif q_position in {31, 14, 30}:
            return k_position == 13
        elif q_position in {15}:
            return k_position == 14
        elif q_position in {16}:
            return k_position == 15
        elif q_position in {17}:
            return k_position == 16
        elif q_position in {18}:
            return k_position == 17
        elif q_position in {19}:
            return k_position == 18
        elif q_position in {20}:
            return k_position == 19
        elif q_position in {21}:
            return k_position == 20
        elif q_position in {22}:
            return k_position == 21
        elif q_position in {23}:
            return k_position == 22
        elif q_position in {24}:
            return k_position == 23
        elif q_position in {25}:
            return k_position == 24
        elif q_position in {26}:
            return k_position == 25
        elif q_position in {27}:
            return k_position == 26
        elif q_position in {28}:
            return k_position == 29
        elif q_position in {29}:
            return k_position == 27

    attn_0_0_pattern = select_closest(positions, positions, predicate_0_0)
    attn_0_0_outputs = aggregate(attn_0_0_pattern, var2_embeddings)
    attn_0_0_output_scores = classifier_weights.loc[
        [("attn_0_0_outputs", str(v)) for v in attn_0_0_outputs]
    ]

    # attn_0_1 ####################################################
    def predicate_0_1(q_position, k_position):
        if q_position in {0, 1, 30, 31}:
            return k_position == 2
        elif q_position in {2}:
            return k_position == 3
        elif q_position in {3, 29}:
            return k_position == 4
        elif q_position in {4}:
            return k_position == 5
        elif q_position in {5}:
            return k_position == 6
        elif q_position in {6}:
            return k_position == 7
        elif q_position in {7}:
            return k_position == 8
        elif q_position in {8}:
            return k_position == 9
        elif q_position in {9}:
            return k_position == 10
        elif q_position in {10}:
            return k_position == 11
        elif q_position in {11}:
            return k_position == 12
        elif q_position in {12}:
            return k_position == 13
        elif q_position in {13}:
            return k_position == 14
        elif q_position in {14}:
            return k_position == 15
        elif q_position in {15}:
            return k_position == 16
        elif q_position in {16}:
            return k_position == 17
        elif q_position in {17}:
            return k_position == 18
        elif q_position in {18}:
            return k_position == 19
        elif q_position in {26, 19}:
            return k_position == 20
        elif q_position in {20}:
            return k_position == 21
        elif q_position in {21}:
            return k_position == 22
        elif q_position in {22, 23}:
            return k_position == 24
        elif q_position in {24, 27}:
            return k_position == 25
        elif q_position in {25}:
            return k_position == 26
        elif q_position in {28}:
            return k_position == 27

    attn_0_1_pattern = select_closest(positions, positions, predicate_0_1)
    attn_0_1_outputs = aggregate(attn_0_1_pattern, var3_embeddings)
    attn_0_1_output_scores = classifier_weights.loc[
        [("attn_0_1_outputs", str(v)) for v in attn_0_1_outputs]
    ]

    # attn_0_2 ####################################################
    def predicate_0_2(q_position, k_position):
        if q_position in {0, 3, 31}:
            return k_position == 2
        elif q_position in {1, 5, 30}:
            return k_position == 4
        elif q_position in {2, 6}:
            return k_position == 5
        elif q_position in {4}:
            return k_position == 3
        elif q_position in {26, 7}:
            return k_position == 6
        elif q_position in {8}:
            return k_position == 7
        elif q_position in {9}:
            return k_position == 8
        elif q_position in {10}:
            return k_position == 9
        elif q_position in {11}:
            return k_position == 10
        elif q_position in {12}:
            return k_position == 11
        elif q_position in {13}:
            return k_position == 12
        elif q_position in {14}:
            return k_position == 13
        elif q_position in {15}:
            return k_position == 14
        elif q_position in {16}:
            return k_position == 15
        elif q_position in {17}:
            return k_position == 16
        elif q_position in {18}:
            return k_position == 17
        elif q_position in {19}:
            return k_position == 18
        elif q_position in {20}:
            return k_position == 19
        elif q_position in {21}:
            return k_position == 20
        elif q_position in {22, 23}:
            return k_position == 21
        elif q_position in {24}:
            return k_position == 23
        elif q_position in {25}:
            return k_position == 24
        elif q_position in {27}:
            return k_position == 26
        elif q_position in {28}:
            return k_position == 27
        elif q_position in {29}:
            return k_position == 28

    attn_0_2_pattern = select_closest(positions, positions, predicate_0_2)
    attn_0_2_outputs = aggregate(attn_0_2_pattern, var0_embeddings)
    attn_0_2_output_scores = classifier_weights.loc[
        [("attn_0_2_outputs", str(v)) for v in attn_0_2_outputs]
    ]

    # attn_0_3 ####################################################
    def predicate_0_3(q_position, k_position):
        if q_position in {0, 3, 31}:
            return k_position == 2
        elif q_position in {1, 5, 30}:
            return k_position == 4
        elif q_position in {2}:
            return k_position == 1
        elif q_position in {4}:
            return k_position == 3
        elif q_position in {6}:
            return k_position == 5
        elif q_position in {7}:
            return k_position == 6
        elif q_position in {8}:
            return k_position == 7
        elif q_position in {9}:
            return k_position == 8
        elif q_position in {10}:
            return k_position == 9
        elif q_position in {11}:
            return k_position == 10
        elif q_position in {12}:
            return k_position == 11
        elif q_position in {13}:
            return k_position == 12
        elif q_position in {14}:
            return k_position == 13
        elif q_position in {15}:
            return k_position == 14
        elif q_position in {16}:
            return k_position == 15
        elif q_position in {17}:
            return k_position == 16
        elif q_position in {18}:
            return k_position == 17
        elif q_position in {19}:
            return k_position == 18
        elif q_position in {20}:
            return k_position == 19
        elif q_position in {21}:
            return k_position == 20
        elif q_position in {22}:
            return k_position == 21
        elif q_position in {23}:
            return k_position == 22
        elif q_position in {24}:
            return k_position == 23
        elif q_position in {25}:
            return k_position == 24
        elif q_position in {26}:
            return k_position == 25
        elif q_position in {27}:
            return k_position == 26
        elif q_position in {28}:
            return k_position == 27
        elif q_position in {29}:
            return k_position == 28

    attn_0_3_pattern = select_closest(positions, positions, predicate_0_3)
    attn_0_3_outputs = aggregate(attn_0_3_pattern, var3_embeddings)
    attn_0_3_output_scores = classifier_weights.loc[
        [("attn_0_3_outputs", str(v)) for v in attn_0_3_outputs]
    ]

    # attn_0_4 ####################################################
    def predicate_0_4(q_position, k_position):
        if q_position in {0}:
            return k_position == 2
        elif q_position in {1, 3, 30, 31}:
            return k_position == 4
        elif q_position in {2}:
            return k_position == 3
        elif q_position in {4}:
            return k_position == 5
        elif q_position in {5}:
            return k_position == 6
        elif q_position in {6}:
            return k_position == 7
        elif q_position in {7}:
            return k_position == 8
        elif q_position in {8}:
            return k_position == 9
        elif q_position in {9}:
            return k_position == 10
        elif q_position in {10}:
            return k_position == 11
        elif q_position in {11}:
            return k_position == 12
        elif q_position in {12}:
            return k_position == 13
        elif q_position in {13}:
            return k_position == 14
        elif q_position in {14}:
            return k_position == 15
        elif q_position in {15}:
            return k_position == 16
        elif q_position in {16}:
            return k_position == 17
        elif q_position in {17, 18}:
            return k_position == 19
        elif q_position in {19}:
            return k_position == 20
        elif q_position in {20}:
            return k_position == 21
        elif q_position in {21}:
            return k_position == 22
        elif q_position in {25, 22, 23}:
            return k_position == 24
        elif q_position in {24}:
            return k_position == 23
        elif q_position in {26}:
            return k_position == 25
        elif q_position in {27, 29}:
            return k_position == 26
        elif q_position in {28}:
            return k_position == 27

    attn_0_4_pattern = select_closest(positions, positions, predicate_0_4)
    attn_0_4_outputs = aggregate(attn_0_4_pattern, var0_embeddings)
    attn_0_4_output_scores = classifier_weights.loc[
        [("attn_0_4_outputs", str(v)) for v in attn_0_4_outputs]
    ]

    # attn_0_5 ####################################################
    def predicate_0_5(q_position, k_position):
        if q_position in {0, 1, 3, 30}:
            return k_position == 2
        elif q_position in {2}:
            return k_position == 1
        elif q_position in {4}:
            return k_position == 3
        elif q_position in {5}:
            return k_position == 4
        elif q_position in {6}:
            return k_position == 5
        elif q_position in {7}:
            return k_position == 6
        elif q_position in {8}:
            return k_position == 7
        elif q_position in {9}:
            return k_position == 8
        elif q_position in {10}:
            return k_position == 9
        elif q_position in {11}:
            return k_position == 10
        elif q_position in {12}:
            return k_position == 11
        elif q_position in {13}:
            return k_position == 12
        elif q_position in {14, 31}:
            return k_position == 13
        elif q_position in {15}:
            return k_position == 14
        elif q_position in {16}:
            return k_position == 15
        elif q_position in {17}:
            return k_position == 16
        elif q_position in {18}:
            return k_position == 17
        elif q_position in {19}:
            return k_position == 18
        elif q_position in {20}:
            return k_position == 19
        elif q_position in {21}:
            return k_position == 20
        elif q_position in {22}:
            return k_position == 21
        elif q_position in {23}:
            return k_position == 22
        elif q_position in {24}:
            return k_position == 23
        elif q_position in {25}:
            return k_position == 24
        elif q_position in {26}:
            return k_position == 25
        elif q_position in {27}:
            return k_position == 26
        elif q_position in {28}:
            return k_position == 27
        elif q_position in {29}:
            return k_position == 28

    attn_0_5_pattern = select_closest(positions, positions, predicate_0_5)
    attn_0_5_outputs = aggregate(attn_0_5_pattern, var1_embeddings)
    attn_0_5_output_scores = classifier_weights.loc[
        [("attn_0_5_outputs", str(v)) for v in attn_0_5_outputs]
    ]

    # attn_0_6 ####################################################
    def predicate_0_6(position, var0_embedding):
        if position in {0, 2, 8, 11, 16, 18, 20, 21, 23, 25, 27, 28, 29, 30}:
            return var0_embedding == Emb0.V04
        elif position in {1, 5}:
            return var0_embedding == Emb0.V13
        elif position in {19, 3}:
            return var0_embedding == Emb0.V15
        elif position in {4}:
            return var0_embedding == Emb0.V05
        elif position in {6, 7}:
            return var0_embedding == Emb0.V25
        elif position in {9, 12}:
            return var0_embedding == Emb0.V11
        elif position in {17, 10}:
            return var0_embedding == Emb0.V14
        elif position in {13}:
            return var0_embedding == Emb0.V12
        elif position in {14}:
            return var0_embedding == Emb0.V08
        elif position in {15}:
            return var0_embedding == Emb0.V20
        elif position in {22}:
            return var0_embedding == Emb0.V21
        elif position in {24}:
            return var0_embedding == Emb0.V09
        elif position in {26}:
            return var0_embedding == Emb0.V24
        elif position in {31}:
            return var0_embedding == Emb0.V02

    attn_0_6_pattern = select_closest(var0_embeddings, positions, predicate_0_6)
    attn_0_6_outputs = aggregate(attn_0_6_pattern, var0_embeddings)
    attn_0_6_output_scores = classifier_weights.loc[
        [("attn_0_6_outputs", str(v)) for v in attn_0_6_outputs]
    ]

    # attn_0_7 ####################################################
    def predicate_0_7(q_position, k_position):
        if q_position in {0, 5, 26, 30, 31}:
            return k_position == 4
        elif q_position in {1, 4, 6}:
            return k_position == 3
        elif q_position in {2}:
            return k_position == 1
        elif q_position in {24, 3}:
            return k_position == 2
        elif q_position in {14, 7}:
            return k_position == 13
        elif q_position in {8}:
            return k_position == 7
        elif q_position in {9}:
            return k_position == 8
        elif q_position in {10}:
            return k_position == 9
        elif q_position in {11}:
            return k_position == 10
        elif q_position in {12}:
            return k_position == 11
        elif q_position in {27, 13}:
            return k_position == 12
        elif q_position in {16, 17, 15}:
            return k_position == 14
        elif q_position in {18}:
            return k_position == 17
        elif q_position in {19, 20}:
            return k_position == 18
        elif q_position in {21}:
            return k_position == 20
        elif q_position in {22}:
            return k_position == 21
        elif q_position in {23}:
            return k_position == 22
        elif q_position in {25, 29}:
            return k_position == 28
        elif q_position in {28}:
            return k_position == 27

    attn_0_7_pattern = select_closest(positions, positions, predicate_0_7)
    attn_0_7_outputs = aggregate(attn_0_7_pattern, var0_embeddings)
    attn_0_7_output_scores = classifier_weights.loc[
        [("attn_0_7_outputs", str(v)) for v in attn_0_7_outputs]
    ]

    # mlp_0_0 #####################################################
    def mlp_0_0(var0_embedding, var3_embedding):
        key = (var0_embedding, var3_embedding)
        if key in {
            (Emb0.V00, Emb3.V00),
            (Emb0.V00, Emb3.V18),
            (Emb0.V00, Emb3.V23),
            (Emb0.V01, Emb3.V00),
            (Emb0.V01, Emb3.V02),
            (Emb0.V01, Emb3.V03),
            (Emb0.V01, Emb3.V04),
            (Emb0.V01, Emb3.V05),
            (Emb0.V01, Emb3.V06),
            (Emb0.V01, Emb3.V09),
            (Emb0.V01, Emb3.V11),
            (Emb0.V01, Emb3.V12),
            (Emb0.V01, Emb3.V15),
            (Emb0.V01, Emb3.V16),
            (Emb0.V01, Emb3.V17),
            (Emb0.V01, Emb3.V18),
            (Emb0.V01, Emb3.V20),
            (Emb0.V01, Emb3.V23),
            (Emb0.V01, Emb3.V27),
            (Emb0.V01, Emb3.V28),
            (Emb0.V01, Emb3.V29),
            (Emb0.V01, Emb3.V31),
            (Emb0.V02, Emb3.V00),
            (Emb0.V02, Emb3.V15),
            (Emb0.V02, Emb3.V18),
            (Emb0.V02, Emb3.V23),
            (Emb0.V03, Emb3.V00),
            (Emb0.V03, Emb3.V15),
            (Emb0.V03, Emb3.V18),
            (Emb0.V03, Emb3.V23),
            (Emb0.V03, Emb3.V28),
            (Emb0.V03, Emb3.V31),
            (Emb0.V04, Emb3.V00),
            (Emb0.V04, Emb3.V02),
            (Emb0.V04, Emb3.V04),
            (Emb0.V04, Emb3.V05),
            (Emb0.V04, Emb3.V06),
            (Emb0.V04, Emb3.V15),
            (Emb0.V04, Emb3.V18),
            (Emb0.V04, Emb3.V23),
            (Emb0.V04, Emb3.V27),
            (Emb0.V04, Emb3.V28),
            (Emb0.V04, Emb3.V29),
            (Emb0.V04, Emb3.V31),
            (Emb0.V05, Emb3.V18),
            (Emb0.V05, Emb3.V23),
            (Emb0.V06, Emb3.V00),
            (Emb0.V06, Emb3.V05),
            (Emb0.V06, Emb3.V15),
            (Emb0.V06, Emb3.V18),
            (Emb0.V06, Emb3.V23),
            (Emb0.V06, Emb3.V27),
            (Emb0.V06, Emb3.V28),
            (Emb0.V06, Emb3.V31),
            (Emb0.V07, Emb3.V00),
            (Emb0.V07, Emb3.V15),
            (Emb0.V07, Emb3.V18),
            (Emb0.V07, Emb3.V23),
            (Emb0.V07, Emb3.V28),
            (Emb0.V08, Emb3.V15),
            (Emb0.V08, Emb3.V18),
            (Emb0.V08, Emb3.V23),
            (Emb0.V09, Emb3.V23),
            (Emb0.V10, Emb3.V00),
            (Emb0.V10, Emb3.V18),
            (Emb0.V10, Emb3.V23),
            (Emb0.V11, Emb3.V00),
            (Emb0.V11, Emb3.V05),
            (Emb0.V11, Emb3.V18),
            (Emb0.V11, Emb3.V23),
            (Emb0.V11, Emb3.V27),
            (Emb0.V12, Emb3.V00),
            (Emb0.V12, Emb3.V01),
            (Emb0.V12, Emb3.V02),
            (Emb0.V12, Emb3.V03),
            (Emb0.V12, Emb3.V04),
            (Emb0.V12, Emb3.V05),
            (Emb0.V12, Emb3.V06),
            (Emb0.V12, Emb3.V09),
            (Emb0.V12, Emb3.V10),
            (Emb0.V12, Emb3.V11),
            (Emb0.V12, Emb3.V12),
            (Emb0.V12, Emb3.V15),
            (Emb0.V12, Emb3.V16),
            (Emb0.V12, Emb3.V17),
            (Emb0.V12, Emb3.V18),
            (Emb0.V12, Emb3.V20),
            (Emb0.V12, Emb3.V21),
            (Emb0.V12, Emb3.V22),
            (Emb0.V12, Emb3.V23),
            (Emb0.V12, Emb3.V24),
            (Emb0.V12, Emb3.V27),
            (Emb0.V12, Emb3.V28),
            (Emb0.V12, Emb3.V29),
            (Emb0.V12, Emb3.V31),
            (Emb0.V13, Emb3.V00),
            (Emb0.V13, Emb3.V18),
            (Emb0.V13, Emb3.V23),
            (Emb0.V14, Emb3.V23),
            (Emb0.V15, Emb3.V00),
            (Emb0.V15, Emb3.V05),
            (Emb0.V15, Emb3.V06),
            (Emb0.V15, Emb3.V15),
            (Emb0.V15, Emb3.V18),
            (Emb0.V15, Emb3.V23),
            (Emb0.V15, Emb3.V27),
            (Emb0.V15, Emb3.V28),
            (Emb0.V15, Emb3.V31),
            (Emb0.V16, Emb3.V00),
            (Emb0.V16, Emb3.V05),
            (Emb0.V16, Emb3.V12),
            (Emb0.V16, Emb3.V18),
            (Emb0.V16, Emb3.V23),
            (Emb0.V16, Emb3.V27),
            (Emb0.V17, Emb3.V00),
            (Emb0.V17, Emb3.V04),
            (Emb0.V17, Emb3.V05),
            (Emb0.V17, Emb3.V06),
            (Emb0.V17, Emb3.V10),
            (Emb0.V17, Emb3.V12),
            (Emb0.V17, Emb3.V15),
            (Emb0.V17, Emb3.V17),
            (Emb0.V17, Emb3.V18),
            (Emb0.V17, Emb3.V23),
            (Emb0.V17, Emb3.V27),
            (Emb0.V17, Emb3.V28),
            (Emb0.V17, Emb3.V31),
            (Emb0.V18, Emb3.V00),
            (Emb0.V18, Emb3.V01),
            (Emb0.V18, Emb3.V02),
            (Emb0.V18, Emb3.V03),
            (Emb0.V18, Emb3.V04),
            (Emb0.V18, Emb3.V05),
            (Emb0.V18, Emb3.V06),
            (Emb0.V18, Emb3.V08),
            (Emb0.V18, Emb3.V09),
            (Emb0.V18, Emb3.V10),
            (Emb0.V18, Emb3.V11),
            (Emb0.V18, Emb3.V12),
            (Emb0.V18, Emb3.V13),
            (Emb0.V18, Emb3.V14),
            (Emb0.V18, Emb3.V15),
            (Emb0.V18, Emb3.V16),
            (Emb0.V18, Emb3.V17),
            (Emb0.V18, Emb3.V18),
            (Emb0.V18, Emb3.V20),
            (Emb0.V18, Emb3.V21),
            (Emb0.V18, Emb3.V22),
            (Emb0.V18, Emb3.V23),
            (Emb0.V18, Emb3.V24),
            (Emb0.V18, Emb3.V26),
            (Emb0.V18, Emb3.V27),
            (Emb0.V18, Emb3.V28),
            (Emb0.V18, Emb3.V29),
            (Emb0.V18, Emb3.V31),
            (Emb0.V19, Emb3.V00),
            (Emb0.V19, Emb3.V04),
            (Emb0.V19, Emb3.V05),
            (Emb0.V19, Emb3.V06),
            (Emb0.V19, Emb3.V15),
            (Emb0.V19, Emb3.V18),
            (Emb0.V19, Emb3.V23),
            (Emb0.V19, Emb3.V27),
            (Emb0.V19, Emb3.V28),
            (Emb0.V20, Emb3.V00),
            (Emb0.V20, Emb3.V05),
            (Emb0.V20, Emb3.V18),
            (Emb0.V20, Emb3.V23),
            (Emb0.V20, Emb3.V27),
            (Emb0.V21, Emb3.V00),
            (Emb0.V21, Emb3.V05),
            (Emb0.V21, Emb3.V18),
            (Emb0.V21, Emb3.V23),
            (Emb0.V22, Emb3.V23),
            (Emb0.V23, Emb3.V00),
            (Emb0.V23, Emb3.V23),
            (Emb0.V24, Emb3.V00),
            (Emb0.V25, Emb3.V00),
            (Emb0.V25, Emb3.V05),
            (Emb0.V25, Emb3.V15),
            (Emb0.V25, Emb3.V18),
            (Emb0.V25, Emb3.V23),
            (Emb0.V25, Emb3.V27),
            (Emb0.V25, Emb3.V28),
            (Emb0.V26, Emb3.V00),
            (Emb0.V26, Emb3.V18),
            (Emb0.V26, Emb3.V23),
            (Emb0.V27, Emb3.V00),
            (Emb0.V27, Emb3.V04),
            (Emb0.V27, Emb3.V05),
            (Emb0.V27, Emb3.V06),
            (Emb0.V27, Emb3.V12),
            (Emb0.V27, Emb3.V15),
            (Emb0.V27, Emb3.V17),
            (Emb0.V27, Emb3.V18),
            (Emb0.V27, Emb3.V23),
            (Emb0.V27, Emb3.V27),
            (Emb0.V27, Emb3.V28),
            (Emb0.V27, Emb3.V29),
            (Emb0.V27, Emb3.V31),
            (Emb0.V28, Emb3.V00),
            (Emb0.V28, Emb3.V02),
            (Emb0.V28, Emb3.V03),
            (Emb0.V28, Emb3.V04),
            (Emb0.V28, Emb3.V05),
            (Emb0.V28, Emb3.V06),
            (Emb0.V28, Emb3.V09),
            (Emb0.V28, Emb3.V10),
            (Emb0.V28, Emb3.V11),
            (Emb0.V28, Emb3.V12),
            (Emb0.V28, Emb3.V13),
            (Emb0.V28, Emb3.V14),
            (Emb0.V28, Emb3.V15),
            (Emb0.V28, Emb3.V17),
            (Emb0.V28, Emb3.V18),
            (Emb0.V28, Emb3.V23),
            (Emb0.V28, Emb3.V27),
            (Emb0.V28, Emb3.V28),
            (Emb0.V28, Emb3.V29),
            (Emb0.V28, Emb3.V31),
            (Emb0.V29, Emb3.V00),
            (Emb0.V29, Emb3.V02),
            (Emb0.V29, Emb3.V03),
            (Emb0.V29, Emb3.V04),
            (Emb0.V29, Emb3.V05),
            (Emb0.V29, Emb3.V06),
            (Emb0.V29, Emb3.V08),
            (Emb0.V29, Emb3.V09),
            (Emb0.V29, Emb3.V10),
            (Emb0.V29, Emb3.V11),
            (Emb0.V29, Emb3.V12),
            (Emb0.V29, Emb3.V13),
            (Emb0.V29, Emb3.V14),
            (Emb0.V29, Emb3.V15),
            (Emb0.V29, Emb3.V16),
            (Emb0.V29, Emb3.V17),
            (Emb0.V29, Emb3.V18),
            (Emb0.V29, Emb3.V20),
            (Emb0.V29, Emb3.V21),
            (Emb0.V29, Emb3.V22),
            (Emb0.V29, Emb3.V23),
            (Emb0.V29, Emb3.V27),
            (Emb0.V29, Emb3.V28),
            (Emb0.V29, Emb3.V29),
            (Emb0.V29, Emb3.V31),
            (Emb0.V30, Emb3.V00),
            (Emb0.V30, Emb3.V02),
            (Emb0.V30, Emb3.V03),
            (Emb0.V30, Emb3.V04),
            (Emb0.V30, Emb3.V05),
            (Emb0.V30, Emb3.V06),
            (Emb0.V30, Emb3.V09),
            (Emb0.V30, Emb3.V11),
            (Emb0.V30, Emb3.V12),
            (Emb0.V30, Emb3.V15),
            (Emb0.V30, Emb3.V16),
            (Emb0.V30, Emb3.V17),
            (Emb0.V30, Emb3.V18),
            (Emb0.V30, Emb3.V20),
            (Emb0.V30, Emb3.V21),
            (Emb0.V30, Emb3.V22),
            (Emb0.V30, Emb3.V23),
            (Emb0.V30, Emb3.V24),
            (Emb0.V30, Emb3.V27),
            (Emb0.V30, Emb3.V28),
            (Emb0.V30, Emb3.V29),
            (Emb0.V30, Emb3.V31),
            (Emb0.V31, Emb3.V00),
            (Emb0.V31, Emb3.V02),
            (Emb0.V31, Emb3.V03),
            (Emb0.V31, Emb3.V04),
            (Emb0.V31, Emb3.V05),
            (Emb0.V31, Emb3.V06),
            (Emb0.V31, Emb3.V09),
            (Emb0.V31, Emb3.V10),
            (Emb0.V31, Emb3.V11),
            (Emb0.V31, Emb3.V12),
            (Emb0.V31, Emb3.V15),
            (Emb0.V31, Emb3.V16),
            (Emb0.V31, Emb3.V17),
            (Emb0.V31, Emb3.V18),
            (Emb0.V31, Emb3.V20),
            (Emb0.V31, Emb3.V23),
            (Emb0.V31, Emb3.V27),
            (Emb0.V31, Emb3.V28),
            (Emb0.V31, Emb3.V29),
            (Emb0.V31, Emb3.V31),
        }:
            return 30
        elif key in {
            (Emb0.V01, Emb3.V01),
            (Emb0.V01, Emb3.V24),
            (Emb0.V02, Emb3.V28),
            (Emb0.V03, Emb3.V24),
            (Emb0.V05, Emb3.V01),
            (Emb0.V05, Emb3.V15),
            (Emb0.V05, Emb3.V24),
            (Emb0.V05, Emb3.V28),
            (Emb0.V06, Emb3.V01),
            (Emb0.V08, Emb3.V24),
            (Emb0.V08, Emb3.V28),
            (Emb0.V09, Emb3.V28),
            (Emb0.V10, Emb3.V01),
            (Emb0.V10, Emb3.V15),
            (Emb0.V10, Emb3.V28),
            (Emb0.V11, Emb3.V01),
            (Emb0.V11, Emb3.V04),
            (Emb0.V11, Emb3.V15),
            (Emb0.V11, Emb3.V24),
            (Emb0.V11, Emb3.V28),
            (Emb0.V11, Emb3.V29),
            (Emb0.V14, Emb3.V28),
            (Emb0.V16, Emb3.V01),
            (Emb0.V16, Emb3.V04),
            (Emb0.V16, Emb3.V06),
            (Emb0.V16, Emb3.V08),
            (Emb0.V16, Emb3.V09),
            (Emb0.V16, Emb3.V11),
            (Emb0.V16, Emb3.V15),
            (Emb0.V16, Emb3.V20),
            (Emb0.V16, Emb3.V28),
            (Emb0.V17, Emb3.V01),
            (Emb0.V17, Emb3.V02),
            (Emb0.V17, Emb3.V08),
            (Emb0.V17, Emb3.V09),
            (Emb0.V17, Emb3.V11),
            (Emb0.V17, Emb3.V20),
            (Emb0.V17, Emb3.V22),
            (Emb0.V17, Emb3.V24),
            (Emb0.V17, Emb3.V25),
            (Emb0.V17, Emb3.V26),
            (Emb0.V17, Emb3.V29),
            (Emb0.V20, Emb3.V01),
            (Emb0.V20, Emb3.V04),
            (Emb0.V20, Emb3.V06),
            (Emb0.V20, Emb3.V15),
            (Emb0.V20, Emb3.V22),
            (Emb0.V20, Emb3.V24),
            (Emb0.V20, Emb3.V28),
            (Emb0.V22, Emb3.V15),
            (Emb0.V23, Emb3.V01),
            (Emb0.V23, Emb3.V02),
            (Emb0.V23, Emb3.V04),
            (Emb0.V23, Emb3.V05),
            (Emb0.V23, Emb3.V06),
            (Emb0.V23, Emb3.V11),
            (Emb0.V23, Emb3.V15),
            (Emb0.V23, Emb3.V16),
            (Emb0.V23, Emb3.V18),
            (Emb0.V23, Emb3.V20),
            (Emb0.V23, Emb3.V22),
            (Emb0.V23, Emb3.V24),
            (Emb0.V23, Emb3.V28),
            (Emb0.V23, Emb3.V29),
            (Emb0.V23, Emb3.V31),
            (Emb0.V24, Emb3.V01),
            (Emb0.V24, Emb3.V04),
            (Emb0.V24, Emb3.V05),
            (Emb0.V24, Emb3.V06),
            (Emb0.V24, Emb3.V08),
            (Emb0.V24, Emb3.V11),
            (Emb0.V24, Emb3.V12),
            (Emb0.V24, Emb3.V15),
            (Emb0.V24, Emb3.V18),
            (Emb0.V24, Emb3.V20),
            (Emb0.V24, Emb3.V22),
            (Emb0.V24, Emb3.V23),
            (Emb0.V24, Emb3.V27),
            (Emb0.V24, Emb3.V28),
            (Emb0.V24, Emb3.V29),
            (Emb0.V25, Emb3.V01),
            (Emb0.V26, Emb3.V01),
            (Emb0.V26, Emb3.V15),
            (Emb0.V26, Emb3.V24),
            (Emb0.V26, Emb3.V28),
            (Emb0.V27, Emb3.V01),
            (Emb0.V28, Emb3.V01),
            (Emb0.V28, Emb3.V08),
            (Emb0.V28, Emb3.V20),
            (Emb0.V28, Emb3.V22),
            (Emb0.V28, Emb3.V24),
            (Emb0.V28, Emb3.V25),
            (Emb0.V29, Emb3.V01),
            (Emb0.V30, Emb3.V01),
            (Emb0.V31, Emb3.V01),
        }:
            return 22
        elif key in {
            (Emb0.V00, Emb3.V16),
            (Emb0.V02, Emb3.V16),
            (Emb0.V02, Emb3.V24),
            (Emb0.V03, Emb3.V16),
            (Emb0.V04, Emb3.V16),
            (Emb0.V05, Emb3.V16),
            (Emb0.V06, Emb3.V16),
            (Emb0.V06, Emb3.V24),
            (Emb0.V06, Emb3.V29),
            (Emb0.V07, Emb3.V16),
            (Emb0.V07, Emb3.V24),
            (Emb0.V07, Emb3.V29),
            (Emb0.V08, Emb3.V16),
            (Emb0.V10, Emb3.V16),
            (Emb0.V10, Emb3.V24),
            (Emb0.V10, Emb3.V29),
            (Emb0.V10, Emb3.V31),
            (Emb0.V11, Emb3.V16),
            (Emb0.V11, Emb3.V31),
            (Emb0.V13, Emb3.V16),
            (Emb0.V14, Emb3.V16),
            (Emb0.V14, Emb3.V24),
            (Emb0.V15, Emb3.V16),
            (Emb0.V15, Emb3.V24),
            (Emb0.V16, Emb3.V16),
            (Emb0.V16, Emb3.V31),
            (Emb0.V17, Emb3.V03),
            (Emb0.V17, Emb3.V16),
            (Emb0.V17, Emb3.V21),
            (Emb0.V19, Emb3.V01),
            (Emb0.V19, Emb3.V02),
            (Emb0.V19, Emb3.V16),
            (Emb0.V19, Emb3.V24),
            (Emb0.V19, Emb3.V29),
            (Emb0.V19, Emb3.V31),
            (Emb0.V20, Emb3.V16),
            (Emb0.V20, Emb3.V29),
            (Emb0.V20, Emb3.V31),
            (Emb0.V21, Emb3.V01),
            (Emb0.V21, Emb3.V02),
            (Emb0.V21, Emb3.V03),
            (Emb0.V21, Emb3.V04),
            (Emb0.V21, Emb3.V06),
            (Emb0.V21, Emb3.V11),
            (Emb0.V21, Emb3.V15),
            (Emb0.V21, Emb3.V16),
            (Emb0.V21, Emb3.V17),
            (Emb0.V21, Emb3.V20),
            (Emb0.V21, Emb3.V21),
            (Emb0.V21, Emb3.V24),
            (Emb0.V21, Emb3.V28),
            (Emb0.V21, Emb3.V29),
            (Emb0.V21, Emb3.V31),
            (Emb0.V22, Emb3.V16),
            (Emb0.V24, Emb3.V02),
            (Emb0.V24, Emb3.V03),
            (Emb0.V24, Emb3.V16),
            (Emb0.V24, Emb3.V17),
            (Emb0.V24, Emb3.V21),
            (Emb0.V24, Emb3.V24),
            (Emb0.V24, Emb3.V31),
            (Emb0.V25, Emb3.V16),
            (Emb0.V25, Emb3.V31),
            (Emb0.V26, Emb3.V16),
            (Emb0.V26, Emb3.V31),
            (Emb0.V27, Emb3.V16),
            (Emb0.V27, Emb3.V24),
            (Emb0.V28, Emb3.V16),
            (Emb0.V28, Emb3.V21),
            (Emb0.V29, Emb3.V24),
        }:
            return 19
        return 29

    mlp_0_0_outputs = [
        mlp_0_0(k0, k1) for k0, k1 in zip(var0_embeddings, var3_embeddings)
    ]
    mlp_0_0_output_scores = classifier_weights.loc[
        [("mlp_0_0_outputs", str(v)) for v in mlp_0_0_outputs]
    ]

    # attn_1_0 ####################################################
    def predicate_1_0(q_position, k_position):
        if q_position in {0, 3, 30, 31}:
            return k_position == 2
        elif q_position in {1, 5}:
            return k_position == 4
        elif q_position in {2}:
            return k_position == 1
        elif q_position in {4, 7}:
            return k_position == 3
        elif q_position in {6}:
            return k_position == 5
        elif q_position in {8, 10}:
            return k_position == 9
        elif q_position in {9}:
            return k_position == 8
        elif q_position in {11}:
            return k_position == 10
        elif q_position in {12, 13}:
            return k_position == 11
        elif q_position in {14}:
            return k_position == 13
        elif q_position in {15}:
            return k_position == 14
        elif q_position in {16}:
            return k_position == 15
        elif q_position in {17}:
            return k_position == 16
        elif q_position in {18}:
            return k_position == 17
        elif q_position in {19}:
            return k_position == 18
        elif q_position in {20}:
            return k_position == 19
        elif q_position in {21}:
            return k_position == 20
        elif q_position in {22}:
            return k_position == 21
        elif q_position in {23}:
            return k_position == 22
        elif q_position in {24}:
            return k_position == 23
        elif q_position in {25, 26}:
            return k_position == 24
        elif q_position in {27}:
            return k_position == 26
        elif q_position in {28}:
            return k_position == 27
        elif q_position in {29}:
            return k_position == 28

    attn_1_0_pattern = select_closest(positions, positions, predicate_1_0)
    attn_1_0_outputs = aggregate(attn_1_0_pattern, attn_0_7_outputs)
    attn_1_0_output_scores = classifier_weights.loc[
        [("attn_1_0_outputs", str(v)) for v in attn_1_0_outputs]
    ]

    # attn_1_1 ####################################################
    def predicate_1_1(q_position, k_position):
        if q_position in {0, 1, 3, 30, 31}:
            return k_position == 2
        elif q_position in {2}:
            return k_position == 1
        elif q_position in {4}:
            return k_position == 3
        elif q_position in {5}:
            return k_position == 4
        elif q_position in {6}:
            return k_position == 5
        elif q_position in {7}:
            return k_position == 6
        elif q_position in {8}:
            return k_position == 7
        elif q_position in {9}:
            return k_position == 8
        elif q_position in {10}:
            return k_position == 9
        elif q_position in {11}:
            return k_position == 10
        elif q_position in {12}:
            return k_position == 11
        elif q_position in {13}:
            return k_position == 12
        elif q_position in {14}:
            return k_position == 13
        elif q_position in {15}:
            return k_position == 14
        elif q_position in {16}:
            return k_position == 15
        elif q_position in {17}:
            return k_position == 16
        elif q_position in {18}:
            return k_position == 17
        elif q_position in {19}:
            return k_position == 18
        elif q_position in {20}:
            return k_position == 19
        elif q_position in {21}:
            return k_position == 20
        elif q_position in {22}:
            return k_position == 21
        elif q_position in {23}:
            return k_position == 22
        elif q_position in {24}:
            return k_position == 23
        elif q_position in {25}:
            return k_position == 24
        elif q_position in {26}:
            return k_position == 25
        elif q_position in {27}:
            return k_position == 26
        elif q_position in {28}:
            return k_position == 27
        elif q_position in {29}:
            return k_position == 28

    attn_1_1_pattern = select_closest(positions, positions, predicate_1_1)
    attn_1_1_outputs = aggregate(attn_1_1_pattern, var0_embeddings)
    attn_1_1_output_scores = classifier_weights.loc[
        [("attn_1_1_outputs", str(v)) for v in attn_1_1_outputs]
    ]

    # attn_1_2 ####################################################
    def predicate_1_2(q_position, k_position):
        if q_position in {0, 5, 30}:
            return k_position == 4
        elif q_position in {1, 19}:
            return k_position == 18
        elif q_position in {2}:
            return k_position == 1
        elif q_position in {3, 31}:
            return k_position == 2
        elif q_position in {4}:
            return k_position == 3
        elif q_position in {6}:
            return k_position == 5
        elif q_position in {7}:
            return k_position == 6
        elif q_position in {8}:
            return k_position == 7
        elif q_position in {9}:
            return k_position == 8
        elif q_position in {10}:
            return k_position == 9
        elif q_position in {11}:
            return k_position == 10
        elif q_position in {12}:
            return k_position == 11
        elif q_position in {13}:
            return k_position == 12
        elif q_position in {14}:
            return k_position == 13
        elif q_position in {15}:
            return k_position == 14
        elif q_position in {16}:
            return k_position == 15
        elif q_position in {17}:
            return k_position == 16
        elif q_position in {18}:
            return k_position == 17
        elif q_position in {20}:
            return k_position == 19
        elif q_position in {21}:
            return k_position == 20
        elif q_position in {25, 22}:
            return k_position == 24
        elif q_position in {23}:
            return k_position == 22
        elif q_position in {24}:
            return k_position == 23
        elif q_position in {26}:
            return k_position == 25
        elif q_position in {27}:
            return k_position == 26
        elif q_position in {28, 29}:
            return k_position == 27

    attn_1_2_pattern = select_closest(positions, positions, predicate_1_2)
    attn_1_2_outputs = aggregate(attn_1_2_pattern, attn_0_2_outputs)
    attn_1_2_output_scores = classifier_weights.loc[
        [("attn_1_2_outputs", str(v)) for v in attn_1_2_outputs]
    ]

    # attn_1_3 ####################################################
    def predicate_1_3(position, var0_embedding):
        if position in {0, 1, 15, 18, 20, 21, 27, 28, 29, 30, 31}:
            return var0_embedding == Emb0.V04
        elif position in {2, 3, 4}:
            return var0_embedding == Emb0.V05
        elif position in {5}:
            return var0_embedding == Emb0.V26
        elif position in {6}:
            return var0_embedding == Emb0.V15
        elif position in {17, 7}:
            return var0_embedding == Emb0.V14
        elif position in {8, 26, 19, 14}:
            return var0_embedding == Emb0.V25
        elif position in {9, 11}:
            return var0_embedding == Emb0.V20
        elif position in {16, 10}:
            return var0_embedding == Emb0.V12
        elif position in {12}:
            return var0_embedding == Emb0.V11
        elif position in {13}:
            return var0_embedding == Emb0.V08
        elif position in {22}:
            return var0_embedding == Emb0.V07
        elif position in {23}:
            return var0_embedding == Emb0.V24
        elif position in {24, 25}:
            return var0_embedding == Emb0.V09

    attn_1_3_pattern = select_closest(var0_embeddings, positions, predicate_1_3)
    attn_1_3_outputs = aggregate(attn_1_3_pattern, var0_embeddings)
    attn_1_3_output_scores = classifier_weights.loc[
        [("attn_1_3_outputs", str(v)) for v in attn_1_3_outputs]
    ]

    # attn_1_4 ####################################################
    def predicate_1_4(q_position, k_position):
        if q_position in {0, 1, 31}:
            return k_position == 2
        elif q_position in {2}:
            return k_position == 1
        elif q_position in {3, 30}:
            return k_position == 4
        elif q_position in {4, 6}:
            return k_position == 5
        elif q_position in {5}:
            return k_position == 6
        elif q_position in {7}:
            return k_position == 8
        elif q_position in {8}:
            return k_position == 9
        elif q_position in {9}:
            return k_position == 10
        elif q_position in {10}:
            return k_position == 11
        elif q_position in {11}:
            return k_position == 12
        elif q_position in {12}:
            return k_position == 13
        elif q_position in {13, 15}:
            return k_position == 14
        elif q_position in {16, 14}:
            return k_position == 15
        elif q_position in {17}:
            return k_position == 16
        elif q_position in {18, 20}:
            return k_position == 19
        elif q_position in {19}:
            return k_position == 18
        elif q_position in {21}:
            return k_position == 20
        elif q_position in {22}:
            return k_position == 21
        elif q_position in {23}:
            return k_position == 22
        elif q_position in {24}:
            return k_position == 23
        elif q_position in {25}:
            return k_position == 24
        elif q_position in {26}:
            return k_position == 25
        elif q_position in {27}:
            return k_position == 26
        elif q_position in {28}:
            return k_position == 27
        elif q_position in {29}:
            return k_position == 28

    attn_1_4_pattern = select_closest(positions, positions, predicate_1_4)
    attn_1_4_outputs = aggregate(attn_1_4_pattern, var0_embeddings)
    attn_1_4_output_scores = classifier_weights.loc[
        [("attn_1_4_outputs", str(v)) for v in attn_1_4_outputs]
    ]

    # attn_1_5 ####################################################
    def predicate_1_5(q_position, k_position):
        if q_position in {0, 1, 3, 30}:
            return k_position == 4
        elif q_position in {2}:
            return k_position == 1
        elif q_position in {4}:
            return k_position == 5
        elif q_position in {5}:
            return k_position == 6
        elif q_position in {6}:
            return k_position == 7
        elif q_position in {7}:
            return k_position == 8
        elif q_position in {8}:
            return k_position == 9
        elif q_position in {9, 11}:
            return k_position == 10
        elif q_position in {10}:
            return k_position == 14
        elif q_position in {12, 14}:
            return k_position == 13
        elif q_position in {13}:
            return k_position == 12
        elif q_position in {29, 15}:
            return k_position == 16
        elif q_position in {16}:
            return k_position == 15
        elif q_position in {17, 19, 20}:
            return k_position == 18
        elif q_position in {18}:
            return k_position == 19
        elif q_position in {21}:
            return k_position == 20
        elif q_position in {25, 22}:
            return k_position == 24
        elif q_position in {23}:
            return k_position == 22
        elif q_position in {24}:
            return k_position == 23
        elif q_position in {26}:
            return k_position == 25
        elif q_position in {27}:
            return k_position == 26
        elif q_position in {28}:
            return k_position == 27
        elif q_position in {31}:
            return k_position == 2

    attn_1_5_pattern = select_closest(positions, positions, predicate_1_5)
    attn_1_5_outputs = aggregate(attn_1_5_pattern, attn_0_4_outputs)
    attn_1_5_output_scores = classifier_weights.loc[
        [("attn_1_5_outputs", str(v)) for v in attn_1_5_outputs]
    ]

    # attn_1_6 ####################################################
    def predicate_1_6(q_position, k_position):
        if q_position in {0, 1, 5, 30, 31}:
            return k_position == 4
        elif q_position in {2}:
            return k_position == 1
        elif q_position in {3}:
            return k_position == 2
        elif q_position in {4}:
            return k_position == 3
        elif q_position in {6}:
            return k_position == 5
        elif q_position in {7}:
            return k_position == 6
        elif q_position in {8}:
            return k_position == 7
        elif q_position in {9}:
            return k_position == 8
        elif q_position in {10}:
            return k_position == 9
        elif q_position in {11}:
            return k_position == 10
        elif q_position in {12}:
            return k_position == 11
        elif q_position in {13}:
            return k_position == 12
        elif q_position in {14}:
            return k_position == 13
        elif q_position in {15}:
            return k_position == 14
        elif q_position in {16}:
            return k_position == 15
        elif q_position in {17}:
            return k_position == 16
        elif q_position in {18}:
            return k_position == 17
        elif q_position in {19}:
            return k_position == 18
        elif q_position in {20}:
            return k_position == 19
        elif q_position in {21}:
            return k_position == 20
        elif q_position in {22}:
            return k_position == 21
        elif q_position in {23}:
            return k_position == 22
        elif q_position in {24}:
            return k_position == 23
        elif q_position in {25}:
            return k_position == 24
        elif q_position in {26}:
            return k_position == 25
        elif q_position in {27}:
            return k_position == 26
        elif q_position in {28, 29}:
            return k_position == 27

    attn_1_6_pattern = select_closest(positions, positions, predicate_1_6)
    attn_1_6_outputs = aggregate(attn_1_6_pattern, attn_0_3_outputs)
    attn_1_6_output_scores = classifier_weights.loc[
        [("attn_1_6_outputs", str(v)) for v in attn_1_6_outputs]
    ]

    # attn_1_7 ####################################################
    def predicate_1_7(q_position, k_position):
        if q_position in {0, 2, 3, 5}:
            return k_position == 4
        elif q_position in {1, 31}:
            return k_position == 2
        elif q_position in {4, 6}:
            return k_position == 5
        elif q_position in {7}:
            return k_position == 6
        elif q_position in {8}:
            return k_position == 7
        elif q_position in {9}:
            return k_position == 8
        elif q_position in {10}:
            return k_position == 9
        elif q_position in {11}:
            return k_position == 10
        elif q_position in {12}:
            return k_position == 11
        elif q_position in {13}:
            return k_position == 12
        elif q_position in {14}:
            return k_position == 13
        elif q_position in {15}:
            return k_position == 14
        elif q_position in {16}:
            return k_position == 15
        elif q_position in {17}:
            return k_position == 16
        elif q_position in {18, 20, 23}:
            return k_position == 19
        elif q_position in {19, 22}:
            return k_position == 21
        elif q_position in {27, 21}:
            return k_position == 20
        elif q_position in {24}:
            return k_position == 22
        elif q_position in {25}:
            return k_position == 24
        elif q_position in {26}:
            return k_position == 25
        elif q_position in {28, 29}:
            return k_position == 27
        elif q_position in {30}:
            return k_position == 26

    attn_1_7_pattern = select_closest(positions, positions, predicate_1_7)
    attn_1_7_outputs = aggregate(attn_1_7_pattern, attn_0_5_outputs)
    attn_1_7_output_scores = classifier_weights.loc[
        [("attn_1_7_outputs", str(v)) for v in attn_1_7_outputs]
    ]

    # mlp_1_0 #####################################################
    def mlp_1_0(var2_embedding):
        key = var2_embedding
        if key in {
            Emb2.V03,
            Emb2.V08,
            Emb2.V12,
            Emb2.V18,
            Emb2.V19,
            Emb2.V21,
            Emb2.V22,
            Emb2.V26,
            Emb2.V30,
        }:
            return 4
        elif key in {Emb2.V29}:
            return 6
        elif key in {Emb2.V16}:
            return 17
        return 18

    mlp_1_0_outputs = [mlp_1_0(k0) for k0 in var2_embeddings]
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
                attn_0_4_output_scores,
                attn_0_5_output_scores,
                attn_0_6_output_scores,
                attn_0_7_output_scores,
                mlp_0_0_output_scores,
                attn_1_0_output_scores,
                attn_1_1_output_scores,
                attn_1_2_output_scores,
                attn_1_3_output_scores,
                attn_1_4_output_scores,
                attn_1_5_output_scores,
                attn_1_6_output_scores,
                attn_1_7_output_scores,
                mlp_1_0_output_scores,
                one_scores,
            ]
        ]
    )
    logits = feature_logits.groupby(level=0).sum(numeric_only=True).to_numpy()
    classes = classifier_weights.columns.to_numpy()
    predictions = classes[logits.argmax(-1)]
    if tokens[0] == "<s>":
        predictions[0] = "<s>"
    if tokens[-1] == "</s>":
        predictions[-1] = "</s>"
    return predictions.tolist()


examples = [
    (
        [
            "<s>",
            "CRICKET",
            "-",
            "<unk>",
            "TAKE",
            "OVER",
            "AT",
            "TOP",
            "AFTER",
            "<unk>",
            "VICTORY",
            ".",
            "</s>",
        ],
        ["<pad>", "O", "O", "B-ORG", "O", "O", "O", "O", "O", "O", "O", "O", "<pad>"],
    ),
    (["<s>", "LONDON", "@-@-@", "</s>"], ["<pad>", "B-LOC", "O", "<pad>"]),
    (
        [
            "<s>",
            "<unk>",
            "by",
            "@",
            ",",
            "Somerset",
            "got",
            "a",
            "solid",
            "start",
            "to",
            "their",
            "second",
            "innings",
            "before",
            "Simmons",
            "stepped",
            "in",
            "to",
            "<unk>",
            "them",
            "out",
            "for",
            "@",
            ".",
            "</s>",
        ],
        [
            "<pad>",
            "O",
            "O",
            "O",
            "O",
            "B-ORG",
            "O",
            "O",
            "O",
            "O",
            "O",
            "O",
            "O",
            "O",
            "O",
            "B-PER",
            "O",
            "O",
            "O",
            "O",
            "O",
            "O",
            "O",
            "O",
            "O",
            "<pad>",
        ],
    ),
    (
        [
            "<s>",
            "Essex",
            ",",
            "however",
            ",",
            "look",
            "certain",
            "to",
            "<unk>",
            "their",
            "top",
            "spot",
            "after",
            "<unk>",
            "Hussain",
            "and",
            "Peter",
            "Such",
            "gave",
            "them",
            "a",
            "firm",
            "<unk>",
            "on",
            "their",
            "match",
            "against",
            "Yorkshire",
            "at",
            "Headingley",
            ".",
            "</s>",
        ],
        [
            "<pad>",
            "B-ORG",
            "O",
            "O",
            "O",
            "O",
            "O",
            "O",
            "O",
            "O",
            "O",
            "O",
            "O",
            "B-PER",
            "I-PER",
            "O",
            "B-PER",
            "I-PER",
            "O",
            "O",
            "O",
            "O",
            "O",
            "O",
            "O",
            "O",
            "O",
            "B-ORG",
            "O",
            "B-LOC",
            "O",
            "<pad>",
        ],
    ),
    (
        [
            "<s>",
            "He",
            "was",
            "well",
            "backed",
            "by",
            "England",
            "<unk>",
            "Mark",
            "<unk>",
            "who",
            "made",
            "@",
            "as",
            "Surrey",
            "closed",
            "on",
            "@",
            "for",
            "seven",
            ",",
            "a",
            "lead",
            "of",
            "@",
            ".",
            "</s>",
        ],
        [
            "<pad>",
            "O",
            "O",
            "O",
            "O",
            "O",
            "B-LOC",
            "O",
            "B-PER",
            "I-PER",
            "O",
            "O",
            "O",
            "O",
            "B-ORG",
            "O",
            "O",
            "O",
            "O",
            "O",
            "O",
            "O",
            "O",
            "O",
            "O",
            "O",
            "<pad>",
        ],
    ),
    (
        [
            "<s>",
            "After",
            "the",
            "<unk>",
            "of",
            "seeing",
            "the",
            "opening",
            "day",
            "of",
            "their",
            "match",
            "<unk>",
            "affected",
            "by",
            "the",
            "weather",
            ",",
            "Kent",
            "stepped",
            "up",
            "a",
            "<unk>",
            "to",
            "<unk>",
            "Nottinghamshire",
            "for",
            "@",
            ".",
            "</s>",
        ],
        [
            "<pad>",
            "O",
            "O",
            "O",
            "O",
            "O",
            "O",
            "O",
            "O",
            "O",
            "O",
            "O",
            "O",
            "O",
            "O",
            "O",
            "O",
            "O",
            "B-ORG",
            "O",
            "O",
            "O",
            "O",
            "O",
            "O",
            "B-ORG",
            "O",
            "O",
            "O",
            "<pad>",
        ],
    ),
    (
        [
            "<s>",
            "They",
            "were",
            "held",
            "up",
            "by",
            "a",
            "<unk>",
            "@",
            "from",
            "Paul",
            "Johnson",
            "but",
            "<unk>",
            "fast",
            "bowler",
            "Martin",
            "<unk>",
            "took",
            "four",
            "for",
            "@",
            ".",
            "</s>",
        ],
        [
            "<pad>",
            "O",
            "O",
            "O",
            "O",
            "O",
            "O",
            "O",
            "O",
            "O",
            "B-PER",
            "I-PER",
            "O",
            "B-MISC",
            "O",
            "O",
            "B-PER",
            "I-PER",
            "O",
            "O",
            "O",
            "O",
            "O",
            "<pad>",
        ],
    ),
    (
        [
            "<s>",
            "By",
            "<unk>",
            "Kent",
            "had",
            "reached",
            "@",
            "for",
            "three",
            ".",
            "</s>",
        ],
        ["<pad>", "O", "O", "B-ORG", "O", "O", "O", "O", "O", "O", "<pad>"],
    ),
    (
        [
            "<s>",
            "CRICKET",
            "-",
            "ENGLISH",
            "COUNTY",
            "CHAMPIONSHIP",
            "SCORES",
            ".",
            "</s>",
        ],
        ["<pad>", "O", "O", "B-MISC", "I-MISC", "I-MISC", "O", "O", "<pad>"],
    ),
    (["<s>", "LONDON", "@-@-@", "</s>"], ["<pad>", "B-LOC", "O", "<pad>"]),
]
for x, y in examples:
    print(f"x: {x}")
    print(f"y: {y}")
    y_hat = run(x)
    print(f"y_hat: {y_hat}")
    print()
