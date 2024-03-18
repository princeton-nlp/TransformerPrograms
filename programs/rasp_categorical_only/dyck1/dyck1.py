import numpy as np
import pandas as pd


def select_closest(keys, queries, predicate):
    scores = [[False for _ in keys] for _ in queries]
    for i, q in enumerate(queries):
        matches = [j for j, k in enumerate(keys[: i + 1]) if predicate(q, k)]
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
        "programs/rasp_categorical_only/dyck1/dyck1_weights.csv",
        index_col=[0, 1],
        dtype={"feature": str},
    )
    # inputs #####################################################
    token_scores = classifier_weights.loc[[("tokens", str(v)) for v in tokens]]

    positions = list(range(len(tokens)))
    position_scores = classifier_weights.loc[[("positions", str(v)) for v in positions]]

    ones = [1 for _ in range(len(tokens))]
    one_scores = classifier_weights.loc[[("ones", "_") for v in ones]].mul(ones, axis=0)

    # attn_0_0 ####################################################
    def predicate_0_0(q_position, k_position):
        if q_position in {0, 15}:
            return k_position == 15
        elif q_position in {1}:
            return k_position == 1
        elif q_position in {2, 6}:
            return k_position == 4
        elif q_position in {3}:
            return k_position == 2
        elif q_position in {4}:
            return k_position == 3
        elif q_position in {5}:
            return k_position == 5
        elif q_position in {7}:
            return k_position == 6
        elif q_position in {8}:
            return k_position == 7
        elif q_position in {9, 10}:
            return k_position == 9
        elif q_position in {11}:
            return k_position == 11
        elif q_position in {12}:
            return k_position == 10
        elif q_position in {13, 14}:
            return k_position == 13

    attn_0_0_pattern = select_closest(positions, positions, predicate_0_0)
    attn_0_0_outputs = aggregate(attn_0_0_pattern, tokens)
    attn_0_0_output_scores = classifier_weights.loc[
        [("attn_0_0_outputs", str(v)) for v in attn_0_0_outputs]
    ]

    # attn_0_1 ####################################################
    def predicate_0_1(q_position, k_position):
        if q_position in {0, 10, 11}:
            return k_position == 9
        elif q_position in {1}:
            return k_position == 1
        elif q_position in {2}:
            return k_position == 6
        elif q_position in {3}:
            return k_position == 2
        elif q_position in {4}:
            return k_position == 3
        elif q_position in {5}:
            return k_position == 5
        elif q_position in {6}:
            return k_position == 4
        elif q_position in {8, 7}:
            return k_position == 7
        elif q_position in {9, 12}:
            return k_position == 8
        elif q_position in {13}:
            return k_position == 11
        elif q_position in {14, 15}:
            return k_position == 10

    attn_0_1_pattern = select_closest(positions, positions, predicate_0_1)
    attn_0_1_outputs = aggregate(attn_0_1_pattern, tokens)
    attn_0_1_output_scores = classifier_weights.loc[
        [("attn_0_1_outputs", str(v)) for v in attn_0_1_outputs]
    ]

    # attn_0_2 ####################################################
    def predicate_0_2(q_position, k_position):
        if q_position in {0, 9}:
            return k_position == 7
        elif q_position in {1}:
            return k_position == 1
        elif q_position in {2}:
            return k_position == 12
        elif q_position in {3, 4, 5, 6}:
            return k_position == 2
        elif q_position in {7}:
            return k_position == 5
        elif q_position in {8}:
            return k_position == 4
        elif q_position in {10, 11, 12, 14}:
            return k_position == 6
        elif q_position in {13}:
            return k_position == 10
        elif q_position in {15}:
            return k_position == 11

    attn_0_2_pattern = select_closest(positions, positions, predicate_0_2)
    attn_0_2_outputs = aggregate(attn_0_2_pattern, tokens)
    attn_0_2_output_scores = classifier_weights.loc[
        [("attn_0_2_outputs", str(v)) for v in attn_0_2_outputs]
    ]

    # attn_0_3 ####################################################
    def predicate_0_3(q_position, k_position):
        if q_position in {0, 12}:
            return k_position == 9
        elif q_position in {1}:
            return k_position == 1
        elif q_position in {8, 2, 13, 15}:
            return k_position == 6
        elif q_position in {3, 4}:
            return k_position == 3
        elif q_position in {5, 6}:
            return k_position == 5
        elif q_position in {7}:
            return k_position == 15
        elif q_position in {9}:
            return k_position == 10
        elif q_position in {10, 11, 14}:
            return k_position == 8

    attn_0_3_pattern = select_closest(positions, positions, predicate_0_3)
    attn_0_3_outputs = aggregate(attn_0_3_pattern, tokens)
    attn_0_3_output_scores = classifier_weights.loc[
        [("attn_0_3_outputs", str(v)) for v in attn_0_3_outputs]
    ]

    # attn_0_4 ####################################################
    def predicate_0_4(q_position, k_position):
        if q_position in {0, 9, 15}:
            return k_position == 13
        elif q_position in {1, 3}:
            return k_position == 2
        elif q_position in {11, 2, 10, 13}:
            return k_position == 8
        elif q_position in {4}:
            return k_position == 3
        elif q_position in {5, 7}:
            return k_position == 15
        elif q_position in {6}:
            return k_position == 4
        elif q_position in {8}:
            return k_position == 5
        elif q_position in {12}:
            return k_position == 10
        elif q_position in {14}:
            return k_position == 11

    attn_0_4_pattern = select_closest(positions, positions, predicate_0_4)
    attn_0_4_outputs = aggregate(attn_0_4_pattern, tokens)
    attn_0_4_output_scores = classifier_weights.loc[
        [("attn_0_4_outputs", str(v)) for v in attn_0_4_outputs]
    ]

    # attn_0_5 ####################################################
    def predicate_0_5(token, position):
        if token in {"("}:
            return position == 1
        elif token in {")"}:
            return position == 4
        elif token in {"<s>"}:
            return position == 2

    attn_0_5_pattern = select_closest(positions, tokens, predicate_0_5)
    attn_0_5_outputs = aggregate(attn_0_5_pattern, tokens)
    attn_0_5_output_scores = classifier_weights.loc[
        [("attn_0_5_outputs", str(v)) for v in attn_0_5_outputs]
    ]

    # attn_0_6 ####################################################
    def predicate_0_6(q_position, k_position):
        if q_position in {0, 6}:
            return k_position == 5
        elif q_position in {1, 12}:
            return k_position == 11
        elif q_position in {2}:
            return k_position == 1
        elif q_position in {3}:
            return k_position == 3
        elif q_position in {4}:
            return k_position == 2
        elif q_position in {5}:
            return k_position == 4
        elif q_position in {7}:
            return k_position == 7
        elif q_position in {8}:
            return k_position == 6
        elif q_position in {9, 10}:
            return k_position == 8
        elif q_position in {11}:
            return k_position == 10
        elif q_position in {13, 14}:
            return k_position == 12
        elif q_position in {15}:
            return k_position == 14

    attn_0_6_pattern = select_closest(positions, positions, predicate_0_6)
    attn_0_6_outputs = aggregate(attn_0_6_pattern, tokens)
    attn_0_6_output_scores = classifier_weights.loc[
        [("attn_0_6_outputs", str(v)) for v in attn_0_6_outputs]
    ]

    # attn_0_7 ####################################################
    def predicate_0_7(q_position, k_position):
        if q_position in {0}:
            return k_position == 10
        elif q_position in {1, 13, 14}:
            return k_position == 13
        elif q_position in {2}:
            return k_position == 4
        elif q_position in {3}:
            return k_position == 3
        elif q_position in {4}:
            return k_position == 2
        elif q_position in {9, 10, 5}:
            return k_position == 9
        elif q_position in {6}:
            return k_position == 5
        elif q_position in {8, 7}:
            return k_position == 6
        elif q_position in {11, 12}:
            return k_position == 11
        elif q_position in {15}:
            return k_position == 12

    attn_0_7_pattern = select_closest(positions, positions, predicate_0_7)
    attn_0_7_outputs = aggregate(attn_0_7_pattern, tokens)
    attn_0_7_output_scores = classifier_weights.loc[
        [("attn_0_7_outputs", str(v)) for v in attn_0_7_outputs]
    ]

    # mlp_0_0 #####################################################
    def mlp_0_0(token, attn_0_7_output):
        key = (token, attn_0_7_output)
        if key in {(")", "("), (")", ")"), (")", "<s>"), ("<s>", ")")}:
            return 4
        return 1

    mlp_0_0_outputs = [mlp_0_0(k0, k1) for k0, k1 in zip(tokens, attn_0_7_outputs)]
    mlp_0_0_output_scores = classifier_weights.loc[
        [("mlp_0_0_outputs", str(v)) for v in mlp_0_0_outputs]
    ]

    # mlp_0_1 #####################################################
    def mlp_0_1(attn_0_6_output, attn_0_0_output):
        key = (attn_0_6_output, attn_0_0_output)
        if key in {(")", ")"), (")", "<s>"), ("<s>", ")")}:
            return 2
        elif key in {("(", "(")}:
            return 10
        elif key in {("(", ")"), (")", "(")}:
            return 8
        return 1

    mlp_0_1_outputs = [
        mlp_0_1(k0, k1) for k0, k1 in zip(attn_0_6_outputs, attn_0_0_outputs)
    ]
    mlp_0_1_output_scores = classifier_weights.loc[
        [("mlp_0_1_outputs", str(v)) for v in mlp_0_1_outputs]
    ]

    # mlp_0_2 #####################################################
    def mlp_0_2(attn_0_0_output, attn_0_6_output):
        key = (attn_0_0_output, attn_0_6_output)
        if key in {(")", ")"), (")", "<s>"), ("<s>", ")"), ("<s>", "<s>")}:
            return 2
        return 13

    mlp_0_2_outputs = [
        mlp_0_2(k0, k1) for k0, k1 in zip(attn_0_0_outputs, attn_0_6_outputs)
    ]
    mlp_0_2_output_scores = classifier_weights.loc[
        [("mlp_0_2_outputs", str(v)) for v in mlp_0_2_outputs]
    ]

    # mlp_0_3 #####################################################
    def mlp_0_3(attn_0_6_output, attn_0_1_output):
        key = (attn_0_6_output, attn_0_1_output)
        return 1

    mlp_0_3_outputs = [
        mlp_0_3(k0, k1) for k0, k1 in zip(attn_0_6_outputs, attn_0_1_outputs)
    ]
    mlp_0_3_output_scores = classifier_weights.loc[
        [("mlp_0_3_outputs", str(v)) for v in mlp_0_3_outputs]
    ]

    # attn_1_0 ####################################################
    def predicate_1_0(mlp_0_0_output, position):
        if mlp_0_0_output in {0, 8, 9, 12, 13}:
            return position == 11
        elif mlp_0_0_output in {1}:
            return position == 7
        elif mlp_0_0_output in {2, 5, 7}:
            return position == 9
        elif mlp_0_0_output in {11, 3, 14}:
            return position == 1
        elif mlp_0_0_output in {4}:
            return position == 6
        elif mlp_0_0_output in {6}:
            return position == 5
        elif mlp_0_0_output in {10, 15}:
            return position == 13

    attn_1_0_pattern = select_closest(positions, mlp_0_0_outputs, predicate_1_0)
    attn_1_0_outputs = aggregate(attn_1_0_pattern, attn_0_5_outputs)
    attn_1_0_output_scores = classifier_weights.loc[
        [("attn_1_0_outputs", str(v)) for v in attn_1_0_outputs]
    ]

    # attn_1_1 ####################################################
    def predicate_1_1(mlp_0_2_output, position):
        if mlp_0_2_output in {0, 2, 6, 7}:
            return position == 5
        elif mlp_0_2_output in {1, 3, 10, 11, 13}:
            return position == 1
        elif mlp_0_2_output in {4, 12}:
            return position == 9
        elif mlp_0_2_output in {8, 5}:
            return position == 7
        elif mlp_0_2_output in {9, 14, 15}:
            return position == 13

    attn_1_1_pattern = select_closest(positions, mlp_0_2_outputs, predicate_1_1)
    attn_1_1_outputs = aggregate(attn_1_1_pattern, tokens)
    attn_1_1_output_scores = classifier_weights.loc[
        [("attn_1_1_outputs", str(v)) for v in attn_1_1_outputs]
    ]

    # attn_1_2 ####################################################
    def predicate_1_2(mlp_0_2_output, position):
        if mlp_0_2_output in {0, 2, 15}:
            return position == 11
        elif mlp_0_2_output in {1, 13}:
            return position == 1
        elif mlp_0_2_output in {3, 6, 7}:
            return position == 5
        elif mlp_0_2_output in {4}:
            return position == 10
        elif mlp_0_2_output in {5}:
            return position == 9
        elif mlp_0_2_output in {8, 11, 12, 14}:
            return position == 13
        elif mlp_0_2_output in {9}:
            return position == 7
        elif mlp_0_2_output in {10}:
            return position == 3

    attn_1_2_pattern = select_closest(positions, mlp_0_2_outputs, predicate_1_2)
    attn_1_2_outputs = aggregate(attn_1_2_pattern, attn_0_3_outputs)
    attn_1_2_output_scores = classifier_weights.loc[
        [("attn_1_2_outputs", str(v)) for v in attn_1_2_outputs]
    ]

    # attn_1_3 ####################################################
    def predicate_1_3(mlp_0_0_output, position):
        if mlp_0_0_output in {0, 5, 7, 13, 15}:
            return position == 11
        elif mlp_0_0_output in {1}:
            return position == 5
        elif mlp_0_0_output in {2, 3}:
            return position == 9
        elif mlp_0_0_output in {4, 12}:
            return position == 4
        elif mlp_0_0_output in {6}:
            return position == 13
        elif mlp_0_0_output in {8, 10, 11}:
            return position == 3
        elif mlp_0_0_output in {9}:
            return position == 7
        elif mlp_0_0_output in {14}:
            return position == 2

    attn_1_3_pattern = select_closest(positions, mlp_0_0_outputs, predicate_1_3)
    attn_1_3_outputs = aggregate(attn_1_3_pattern, attn_0_2_outputs)
    attn_1_3_output_scores = classifier_weights.loc[
        [("attn_1_3_outputs", str(v)) for v in attn_1_3_outputs]
    ]

    # attn_1_4 ####################################################
    def predicate_1_4(attn_0_0_output, position):
        if attn_0_0_output in {"<s>", "("}:
            return position == 1
        elif attn_0_0_output in {")"}:
            return position == 7

    attn_1_4_pattern = select_closest(positions, attn_0_0_outputs, predicate_1_4)
    attn_1_4_outputs = aggregate(attn_1_4_pattern, tokens)
    attn_1_4_output_scores = classifier_weights.loc[
        [("attn_1_4_outputs", str(v)) for v in attn_1_4_outputs]
    ]

    # attn_1_5 ####################################################
    def predicate_1_5(mlp_0_2_output, mlp_0_1_output):
        if mlp_0_2_output in {0, 1, 8, 10, 11, 14}:
            return mlp_0_1_output == 2
        elif mlp_0_2_output in {2, 13, 7}:
            return mlp_0_1_output == 10
        elif mlp_0_2_output in {9, 3}:
            return mlp_0_1_output == 12
        elif mlp_0_2_output in {4}:
            return mlp_0_1_output == 13
        elif mlp_0_2_output in {5, 15}:
            return mlp_0_1_output == 4
        elif mlp_0_2_output in {6}:
            return mlp_0_1_output == 11
        elif mlp_0_2_output in {12}:
            return mlp_0_1_output == 15

    attn_1_5_pattern = select_closest(mlp_0_1_outputs, mlp_0_2_outputs, predicate_1_5)
    attn_1_5_outputs = aggregate(attn_1_5_pattern, attn_0_6_outputs)
    attn_1_5_output_scores = classifier_weights.loc[
        [("attn_1_5_outputs", str(v)) for v in attn_1_5_outputs]
    ]

    # attn_1_6 ####################################################
    def predicate_1_6(position, mlp_0_1_output):
        if position in {0}:
            return mlp_0_1_output == 5
        elif position in {1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}:
            return mlp_0_1_output == 2
        elif position in {2}:
            return mlp_0_1_output == 7

    attn_1_6_pattern = select_closest(mlp_0_1_outputs, positions, predicate_1_6)
    attn_1_6_outputs = aggregate(attn_1_6_pattern, attn_0_1_outputs)
    attn_1_6_output_scores = classifier_weights.loc[
        [("attn_1_6_outputs", str(v)) for v in attn_1_6_outputs]
    ]

    # attn_1_7 ####################################################
    def predicate_1_7(attn_0_0_output, position):
        if attn_0_0_output in {"<s>", "("}:
            return position == 1
        elif attn_0_0_output in {")"}:
            return position == 9

    attn_1_7_pattern = select_closest(positions, attn_0_0_outputs, predicate_1_7)
    attn_1_7_outputs = aggregate(attn_1_7_pattern, tokens)
    attn_1_7_output_scores = classifier_weights.loc[
        [("attn_1_7_outputs", str(v)) for v in attn_1_7_outputs]
    ]

    # mlp_1_0 #####################################################
    def mlp_1_0(attn_0_0_output, token):
        key = (attn_0_0_output, token)
        if key in {("(", "<s>")}:
            return 12
        elif key in {("(", "(")}:
            return 10
        elif key in {("<s>", ")"), ("<s>", "<s>")}:
            return 5
        return 2

    mlp_1_0_outputs = [mlp_1_0(k0, k1) for k0, k1 in zip(attn_0_0_outputs, tokens)]
    mlp_1_0_output_scores = classifier_weights.loc[
        [("mlp_1_0_outputs", str(v)) for v in mlp_1_0_outputs]
    ]

    # mlp_1_1 #####################################################
    def mlp_1_1(attn_1_5_output, attn_1_7_output):
        key = (attn_1_5_output, attn_1_7_output)
        if key in {("(", "("), ("(", "<s>"), (")", "<s>"), ("<s>", "(")}:
            return 11
        elif key in {("<s>", "<s>")}:
            return 0
        return 9

    mlp_1_1_outputs = [
        mlp_1_1(k0, k1) for k0, k1 in zip(attn_1_5_outputs, attn_1_7_outputs)
    ]
    mlp_1_1_output_scores = classifier_weights.loc[
        [("mlp_1_1_outputs", str(v)) for v in mlp_1_1_outputs]
    ]

    # mlp_1_2 #####################################################
    def mlp_1_2(attn_1_6_output, attn_1_2_output):
        key = (attn_1_6_output, attn_1_2_output)
        if key in {("(", ")"), (")", "<s>"), ("<s>", "<s>")}:
            return 3
        elif key in {(")", ")"), ("<s>", ")")}:
            return 2
        elif key in {("<s>", "(")}:
            return 0
        elif key in {(")", "(")}:
            return 6
        return 4

    mlp_1_2_outputs = [
        mlp_1_2(k0, k1) for k0, k1 in zip(attn_1_6_outputs, attn_1_2_outputs)
    ]
    mlp_1_2_output_scores = classifier_weights.loc[
        [("mlp_1_2_outputs", str(v)) for v in mlp_1_2_outputs]
    ]

    # mlp_1_3 #####################################################
    def mlp_1_3(position):
        key = position
        if key in {7, 10, 11}:
            return 1
        elif key in {3}:
            return 3
        elif key in {5}:
            return 5
        elif key in {15}:
            return 13
        return 2

    mlp_1_3_outputs = [mlp_1_3(k0) for k0 in positions]
    mlp_1_3_output_scores = classifier_weights.loc[
        [("mlp_1_3_outputs", str(v)) for v in mlp_1_3_outputs]
    ]

    # attn_2_0 ####################################################
    def predicate_2_0(attn_0_2_output, position):
        if attn_0_2_output in {"("}:
            return position == 5
        elif attn_0_2_output in {")"}:
            return position == 7
        elif attn_0_2_output in {"<s>"}:
            return position == 1

    attn_2_0_pattern = select_closest(positions, attn_0_2_outputs, predicate_2_0)
    attn_2_0_outputs = aggregate(attn_2_0_pattern, attn_1_1_outputs)
    attn_2_0_output_scores = classifier_weights.loc[
        [("attn_2_0_outputs", str(v)) for v in attn_2_0_outputs]
    ]

    # attn_2_1 ####################################################
    def predicate_2_1(attn_0_2_output, position):
        if attn_0_2_output in {"("}:
            return position == 3
        elif attn_0_2_output in {")"}:
            return position == 7
        elif attn_0_2_output in {"<s>"}:
            return position == 4

    attn_2_1_pattern = select_closest(positions, attn_0_2_outputs, predicate_2_1)
    attn_2_1_outputs = aggregate(attn_2_1_pattern, attn_0_5_outputs)
    attn_2_1_output_scores = classifier_weights.loc[
        [("attn_2_1_outputs", str(v)) for v in attn_2_1_outputs]
    ]

    # attn_2_2 ####################################################
    def predicate_2_2(mlp_0_3_output, mlp_0_1_output):
        if mlp_0_3_output in {0, 9, 15}:
            return mlp_0_1_output == 7
        elif mlp_0_3_output in {1, 11}:
            return mlp_0_1_output == 2
        elif mlp_0_3_output in {2}:
            return mlp_0_1_output == 10
        elif mlp_0_3_output in {3, 4, 5, 6, 8, 12, 14}:
            return mlp_0_1_output == 5
        elif mlp_0_3_output in {7}:
            return mlp_0_1_output == 9
        elif mlp_0_3_output in {10, 13}:
            return mlp_0_1_output == 3

    attn_2_2_pattern = select_closest(mlp_0_1_outputs, mlp_0_3_outputs, predicate_2_2)
    attn_2_2_outputs = aggregate(attn_2_2_pattern, attn_1_6_outputs)
    attn_2_2_output_scores = classifier_weights.loc[
        [("attn_2_2_outputs", str(v)) for v in attn_2_2_outputs]
    ]

    # attn_2_3 ####################################################
    def predicate_2_3(mlp_0_2_output, mlp_0_1_output):
        if mlp_0_2_output in {0}:
            return mlp_0_1_output == 9
        elif mlp_0_2_output in {1, 3, 6, 7, 8, 9, 10, 11, 13, 15}:
            return mlp_0_1_output == 2
        elif mlp_0_2_output in {2}:
            return mlp_0_1_output == 10
        elif mlp_0_2_output in {4, 12}:
            return mlp_0_1_output == 15
        elif mlp_0_2_output in {5}:
            return mlp_0_1_output == 12
        elif mlp_0_2_output in {14}:
            return mlp_0_1_output == 6

    attn_2_3_pattern = select_closest(mlp_0_1_outputs, mlp_0_2_outputs, predicate_2_3)
    attn_2_3_outputs = aggregate(attn_2_3_pattern, mlp_1_1_outputs)
    attn_2_3_output_scores = classifier_weights.loc[
        [("attn_2_3_outputs", str(v)) for v in attn_2_3_outputs]
    ]

    # attn_2_4 ####################################################
    def predicate_2_4(mlp_0_0_output, mlp_0_1_output):
        if mlp_0_0_output in {0, 1, 2, 4, 5, 7, 8, 9, 10, 11, 12, 13}:
            return mlp_0_1_output == 2
        elif mlp_0_0_output in {3, 14}:
            return mlp_0_1_output == 15
        elif mlp_0_0_output in {6, 15}:
            return mlp_0_1_output == 9

    attn_2_4_pattern = select_closest(mlp_0_1_outputs, mlp_0_0_outputs, predicate_2_4)
    attn_2_4_outputs = aggregate(attn_2_4_pattern, attn_1_5_outputs)
    attn_2_4_output_scores = classifier_weights.loc[
        [("attn_2_4_outputs", str(v)) for v in attn_2_4_outputs]
    ]

    # attn_2_5 ####################################################
    def predicate_2_5(attn_0_2_output, position):
        if attn_0_2_output in {"("}:
            return position == 3
        elif attn_0_2_output in {")"}:
            return position == 5
        elif attn_0_2_output in {"<s>"}:
            return position == 1

    attn_2_5_pattern = select_closest(positions, attn_0_2_outputs, predicate_2_5)
    attn_2_5_outputs = aggregate(attn_2_5_pattern, mlp_1_2_outputs)
    attn_2_5_output_scores = classifier_weights.loc[
        [("attn_2_5_outputs", str(v)) for v in attn_2_5_outputs]
    ]

    # attn_2_6 ####################################################
    def predicate_2_6(attn_0_2_output, position):
        if attn_0_2_output in {"(", ")"}:
            return position == 7
        elif attn_0_2_output in {"<s>"}:
            return position == 1

    attn_2_6_pattern = select_closest(positions, attn_0_2_outputs, predicate_2_6)
    attn_2_6_outputs = aggregate(attn_2_6_pattern, attn_1_2_outputs)
    attn_2_6_output_scores = classifier_weights.loc[
        [("attn_2_6_outputs", str(v)) for v in attn_2_6_outputs]
    ]

    # attn_2_7 ####################################################
    def predicate_2_7(mlp_0_2_output, position):
        if mlp_0_2_output in {0, 12}:
            return position == 9
        elif mlp_0_2_output in {1, 14}:
            return position == 7
        elif mlp_0_2_output in {11, 2, 10, 13}:
            return position == 4
        elif mlp_0_2_output in {3, 4, 5, 6, 8, 9, 15}:
            return position == 5
        elif mlp_0_2_output in {7}:
            return position == 10

    attn_2_7_pattern = select_closest(positions, mlp_0_2_outputs, predicate_2_7)
    attn_2_7_outputs = aggregate(attn_2_7_pattern, attn_1_4_outputs)
    attn_2_7_output_scores = classifier_weights.loc[
        [("attn_2_7_outputs", str(v)) for v in attn_2_7_outputs]
    ]

    # mlp_2_0 #####################################################
    def mlp_2_0(attn_2_5_output, attn_2_3_output):
        key = (attn_2_5_output, attn_2_3_output)
        return 7

    mlp_2_0_outputs = [
        mlp_2_0(k0, k1) for k0, k1 in zip(attn_2_5_outputs, attn_2_3_outputs)
    ]
    mlp_2_0_output_scores = classifier_weights.loc[
        [("mlp_2_0_outputs", str(v)) for v in mlp_2_0_outputs]
    ]

    # mlp_2_1 #####################################################
    def mlp_2_1(attn_1_6_output, attn_2_6_output):
        key = (attn_1_6_output, attn_2_6_output)
        return 4

    mlp_2_1_outputs = [
        mlp_2_1(k0, k1) for k0, k1 in zip(attn_1_6_outputs, attn_2_6_outputs)
    ]
    mlp_2_1_output_scores = classifier_weights.loc[
        [("mlp_2_1_outputs", str(v)) for v in mlp_2_1_outputs]
    ]

    # mlp_2_2 #####################################################
    def mlp_2_2(attn_2_0_output, attn_2_1_output):
        key = (attn_2_0_output, attn_2_1_output)
        if key in {(")", ")"), (")", "<s>"), ("<s>", ")")}:
            return 1
        elif key in {("(", ")"), (")", "(")}:
            return 15
        return 9

    mlp_2_2_outputs = [
        mlp_2_2(k0, k1) for k0, k1 in zip(attn_2_0_outputs, attn_2_1_outputs)
    ]
    mlp_2_2_output_scores = classifier_weights.loc[
        [("mlp_2_2_outputs", str(v)) for v in mlp_2_2_outputs]
    ]

    # mlp_2_3 #####################################################
    def mlp_2_3(attn_2_4_output, attn_2_3_output):
        key = (attn_2_4_output, attn_2_3_output)
        if key in {
            (")", 2),
            (")", 4),
            (")", 5),
            (")", 7),
            (")", 9),
            (")", 10),
            (")", 12),
            (")", 14),
        }:
            return 14
        elif key in {(")", 1), (")", 3), (")", 8), (")", 11), (")", 13), (")", 15)}:
            return 13
        elif key in {(")", 0), (")", 6)}:
            return 10
        return 15

    mlp_2_3_outputs = [
        mlp_2_3(k0, k1) for k0, k1 in zip(attn_2_4_outputs, attn_2_3_outputs)
    ]
    mlp_2_3_output_scores = classifier_weights.loc[
        [("mlp_2_3_outputs", str(v)) for v in mlp_2_3_outputs]
    ]

    feature_logits = pd.concat(
        [
            df.reset_index()
            for df in [
                token_scores,
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
                mlp_0_1_output_scores,
                mlp_0_2_output_scores,
                mlp_0_3_output_scores,
                attn_1_0_output_scores,
                attn_1_1_output_scores,
                attn_1_2_output_scores,
                attn_1_3_output_scores,
                attn_1_4_output_scores,
                attn_1_5_output_scores,
                attn_1_6_output_scores,
                attn_1_7_output_scores,
                mlp_1_0_output_scores,
                mlp_1_1_output_scores,
                mlp_1_2_output_scores,
                mlp_1_3_output_scores,
                attn_2_0_output_scores,
                attn_2_1_output_scores,
                attn_2_2_output_scores,
                attn_2_3_output_scores,
                attn_2_4_output_scores,
                attn_2_5_output_scores,
                attn_2_6_output_scores,
                attn_2_7_output_scores,
                mlp_2_0_output_scores,
                mlp_2_1_output_scores,
                mlp_2_2_output_scores,
                mlp_2_3_output_scores,
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
            ")",
            ")",
            "(",
            "(",
            ")",
            ")",
            ")",
            "(",
            ")",
            "(",
            "(",
            ")",
            "(",
            ")",
            ")",
        ],
        [
            "<pad>",
            "F",
            "F",
            "F",
            "F",
            "F",
            "F",
            "F",
            "F",
            "F",
            "F",
            "F",
            "F",
            "F",
            "F",
            "F",
        ],
    ),
    (
        [
            "<s>",
            "(",
            ")",
            "(",
            "(",
            "(",
            "(",
            ")",
            ")",
            ")",
            ")",
            "(",
            ")",
            "(",
            "(",
            "(",
        ],
        [
            "<pad>",
            "P",
            "T",
            "P",
            "P",
            "P",
            "P",
            "P",
            "P",
            "P",
            "T",
            "P",
            "T",
            "P",
            "P",
            "P",
        ],
    ),
    (
        [
            "<s>",
            "(",
            ")",
            ")",
            "(",
            "(",
            "(",
            "(",
            "(",
            ")",
            "(",
            ")",
            "(",
            ")",
            "(",
            "(",
        ],
        [
            "<pad>",
            "P",
            "T",
            "F",
            "F",
            "F",
            "F",
            "F",
            "F",
            "F",
            "F",
            "F",
            "F",
            "F",
            "F",
            "F",
        ],
    ),
    (
        [
            "<s>",
            "(",
            "(",
            "(",
            ")",
            ")",
            ")",
            ")",
            ")",
            ")",
            ")",
            ")",
            ")",
            ")",
            "(",
            "(",
        ],
        [
            "<pad>",
            "P",
            "P",
            "P",
            "P",
            "P",
            "T",
            "F",
            "F",
            "F",
            "F",
            "F",
            "F",
            "F",
            "F",
            "F",
        ],
    ),
    (
        [
            "<s>",
            "(",
            ")",
            ")",
            ")",
            "(",
            ")",
            "(",
            ")",
            "(",
            "(",
            ")",
            ")",
            "(",
            ")",
            "(",
        ],
        [
            "<pad>",
            "P",
            "T",
            "F",
            "F",
            "F",
            "F",
            "F",
            "F",
            "F",
            "F",
            "F",
            "F",
            "F",
            "F",
            "F",
        ],
    ),
    (
        [
            "<s>",
            ")",
            ")",
            "(",
            "(",
            ")",
            "(",
            "(",
            "(",
            ")",
            "(",
            "(",
            "(",
            ")",
            "(",
            "(",
        ],
        [
            "<pad>",
            "F",
            "F",
            "F",
            "F",
            "F",
            "F",
            "F",
            "F",
            "F",
            "F",
            "F",
            "F",
            "F",
            "F",
            "F",
        ],
    ),
    (
        [
            "<s>",
            ")",
            "(",
            ")",
            ")",
            "(",
            ")",
            ")",
            "(",
            "(",
            "(",
            "(",
            "(",
            "(",
            "(",
            ")",
        ],
        [
            "<pad>",
            "F",
            "F",
            "F",
            "F",
            "F",
            "F",
            "F",
            "F",
            "F",
            "F",
            "F",
            "F",
            "F",
            "F",
            "F",
        ],
    ),
    (
        [
            "<s>",
            "(",
            ")",
            "(",
            "(",
            "(",
            "(",
            ")",
            "(",
            "(",
            ")",
            "(",
            "(",
            ")",
            "(",
            "(",
        ],
        [
            "<pad>",
            "P",
            "T",
            "P",
            "P",
            "P",
            "P",
            "P",
            "P",
            "P",
            "P",
            "P",
            "P",
            "P",
            "P",
            "P",
        ],
    ),
    (
        [
            "<s>",
            "(",
            ")",
            ")",
            "(",
            ")",
            ")",
            "(",
            "(",
            "(",
            "(",
            ")",
            "(",
            "(",
            ")",
            "(",
        ],
        [
            "<pad>",
            "P",
            "T",
            "F",
            "F",
            "F",
            "F",
            "F",
            "F",
            "F",
            "F",
            "F",
            "F",
            "F",
            "F",
            "F",
        ],
    ),
    (
        [
            "<s>",
            "(",
            "(",
            "(",
            "(",
            "(",
            "(",
            "(",
            "(",
            ")",
            "(",
            ")",
            "(",
            ")",
            ")",
            ")",
        ],
        [
            "<pad>",
            "P",
            "P",
            "P",
            "P",
            "P",
            "P",
            "P",
            "P",
            "P",
            "P",
            "P",
            "P",
            "P",
            "P",
            "P",
        ],
    ),
]
for x, y in examples:
    print(f"x: {x}")
    print(f"y: {y}")
    y_hat = run(x)
    print(f"y_hat: {y_hat}")
    print()
