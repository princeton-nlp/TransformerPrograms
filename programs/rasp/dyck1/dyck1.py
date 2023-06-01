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


def select(keys, queries, predicate):
    return [[predicate(q, k) for k in keys[: i + 1]] for i, q in enumerate(queries)]


def aggregate(attention, values):
    return [[v for a, v in zip(attn, values) if a][0] for attn in attention]


def aggregate_sum(attention, values):
    return [sum([v for a, v in zip(attn, values) if a]) for attn in attention]


def run(tokens):
    # classifier weights ##########################################
    classifier_weights = pd.read_csv(
        "programs/rasp/dyck1/dyck1_weights.csv",
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
        elif q_position in {8, 2}:
            return k_position == 6
        elif q_position in {13, 3, 5, 14}:
            return k_position == 12
        elif q_position in {4}:
            return k_position == 3
        elif q_position in {6}:
            return k_position == 4
        elif q_position in {7}:
            return k_position == 13
        elif q_position in {9, 10}:
            return k_position == 7
        elif q_position in {11}:
            return k_position == 9
        elif q_position in {12}:
            return k_position == 11

    attn_0_0_pattern = select_closest(positions, positions, predicate_0_0)
    attn_0_0_outputs = aggregate(attn_0_0_pattern, tokens)
    attn_0_0_output_scores = classifier_weights.loc[
        [("attn_0_0_outputs", str(v)) for v in attn_0_0_outputs]
    ]

    # attn_0_1 ####################################################
    def predicate_0_1(token, position):
        if token in {"("}:
            return position == 1
        elif token in {")"}:
            return position == 4
        elif token in {"<s>"}:
            return position == 9

    attn_0_1_pattern = select_closest(positions, tokens, predicate_0_1)
    attn_0_1_outputs = aggregate(attn_0_1_pattern, tokens)
    attn_0_1_output_scores = classifier_weights.loc[
        [("attn_0_1_outputs", str(v)) for v in attn_0_1_outputs]
    ]

    # attn_0_2 ####################################################
    def predicate_0_2(token, position):
        if token in {"("}:
            return position == 1
        elif token in {")"}:
            return position == 4
        elif token in {"<s>"}:
            return position == 11

    attn_0_2_pattern = select_closest(positions, tokens, predicate_0_2)
    attn_0_2_outputs = aggregate(attn_0_2_pattern, tokens)
    attn_0_2_output_scores = classifier_weights.loc[
        [("attn_0_2_outputs", str(v)) for v in attn_0_2_outputs]
    ]

    # attn_0_3 ####################################################
    def predicate_0_3(token, position):
        if token in {"("}:
            return position == 1
        elif token in {")"}:
            return position == 6
        elif token in {"<s>"}:
            return position == 9

    attn_0_3_pattern = select_closest(positions, tokens, predicate_0_3)
    attn_0_3_outputs = aggregate(attn_0_3_pattern, tokens)
    attn_0_3_output_scores = classifier_weights.loc[
        [("attn_0_3_outputs", str(v)) for v in attn_0_3_outputs]
    ]

    # num_attn_0_0 ####################################################
    def num_predicate_0_0(position, token):
        if position in {0, 7, 9, 10, 11, 12, 13, 15}:
            return token == ""
        elif position in {1, 14}:
            return token == "<s>"
        elif position in {2, 3, 4, 5, 6, 8}:
            return token == ")"

    num_attn_0_0_pattern = select(tokens, positions, num_predicate_0_0)
    num_attn_0_0_outputs = aggregate_sum(num_attn_0_0_pattern, ones)
    num_attn_0_0_output_scores = classifier_weights.loc[
        [("num_attn_0_0_outputs", "_") for v in num_attn_0_0_outputs]
    ].mul(num_attn_0_0_outputs, axis=0)

    # num_attn_0_1 ####################################################
    def num_predicate_0_1(q_position, k_position):
        if q_position in {0}:
            return k_position == 11
        elif q_position in {1, 9, 15}:
            return k_position == 3
        elif q_position in {2}:
            return k_position == 15
        elif q_position in {3, 5, 6, 7}:
            return k_position == 2
        elif q_position in {10, 4}:
            return k_position == 9
        elif q_position in {8}:
            return k_position == 8
        elif q_position in {11}:
            return k_position == 4
        elif q_position in {12}:
            return k_position == 12
        elif q_position in {13}:
            return k_position == 5
        elif q_position in {14}:
            return k_position == 6

    num_attn_0_1_pattern = select(positions, positions, num_predicate_0_1)
    num_attn_0_1_outputs = aggregate_sum(num_attn_0_1_pattern, ones)
    num_attn_0_1_output_scores = classifier_weights.loc[
        [("num_attn_0_1_outputs", "_") for v in num_attn_0_1_outputs]
    ].mul(num_attn_0_1_outputs, axis=0)

    # num_attn_0_2 ####################################################
    def num_predicate_0_2(position, token):
        if position in {0, 2, 4, 6, 8, 10, 12, 14}:
            return token == ""
        elif position in {1, 3, 5, 7, 9}:
            return token == ")"
        elif position in {11, 13, 15}:
            return token == "<s>"

    num_attn_0_2_pattern = select(tokens, positions, num_predicate_0_2)
    num_attn_0_2_outputs = aggregate_sum(num_attn_0_2_pattern, ones)
    num_attn_0_2_output_scores = classifier_weights.loc[
        [("num_attn_0_2_outputs", "_") for v in num_attn_0_2_outputs]
    ].mul(num_attn_0_2_outputs, axis=0)

    # num_attn_0_3 ####################################################
    def num_predicate_0_3(q_position, k_position):
        if q_position in {0, 11}:
            return k_position == 4
        elif q_position in {1}:
            return k_position == 5
        elif q_position in {2}:
            return k_position == 11
        elif q_position in {3, 5, 15}:
            return k_position == 2
        elif q_position in {4}:
            return k_position == 12
        elif q_position in {6}:
            return k_position == 10
        elif q_position in {13, 14, 7}:
            return k_position == 3
        elif q_position in {8}:
            return k_position == 15
        elif q_position in {9}:
            return k_position == 13
        elif q_position in {10}:
            return k_position == 8
        elif q_position in {12}:
            return k_position == 0

    num_attn_0_3_pattern = select(positions, positions, num_predicate_0_3)
    num_attn_0_3_outputs = aggregate_sum(num_attn_0_3_pattern, ones)
    num_attn_0_3_output_scores = classifier_weights.loc[
        [("num_attn_0_3_outputs", "_") for v in num_attn_0_3_outputs]
    ].mul(num_attn_0_3_outputs, axis=0)

    # mlp_0_0 #####################################################
    def mlp_0_0(attn_0_1_output, attn_0_0_output):
        key = (attn_0_1_output, attn_0_0_output)
        if key in {("(", ")"), (")", ")"), (")", "<s>"), ("<s>", ")"), ("<s>", "<s>")}:
            return 2
        elif key in {(")", "(")}:
            return 10
        return 9

    mlp_0_0_outputs = [
        mlp_0_0(k0, k1) for k0, k1 in zip(attn_0_1_outputs, attn_0_0_outputs)
    ]
    mlp_0_0_output_scores = classifier_weights.loc[
        [("mlp_0_0_outputs", str(v)) for v in mlp_0_0_outputs]
    ]

    # num_mlp_0_0 #################################################
    def num_mlp_0_0(num_attn_0_0_output):
        key = num_attn_0_0_output
        return 14

    num_mlp_0_0_outputs = [num_mlp_0_0(k0) for k0 in num_attn_0_0_outputs]
    num_mlp_0_0_output_scores = classifier_weights.loc[
        [("num_mlp_0_0_outputs", str(v)) for v in num_mlp_0_0_outputs]
    ]

    # attn_1_0 ####################################################
    def predicate_1_0(token, position):
        if token in {"("}:
            return position == 1
        elif token in {")"}:
            return position == 2
        elif token in {"<s>"}:
            return position == 5

    attn_1_0_pattern = select_closest(positions, tokens, predicate_1_0)
    attn_1_0_outputs = aggregate(attn_1_0_pattern, attn_0_2_outputs)
    attn_1_0_output_scores = classifier_weights.loc[
        [("attn_1_0_outputs", str(v)) for v in attn_1_0_outputs]
    ]

    # attn_1_1 ####################################################
    def predicate_1_1(token, position):
        if token in {"("}:
            return position == 1
        elif token in {")"}:
            return position == 8
        elif token in {"<s>"}:
            return position == 5

    attn_1_1_pattern = select_closest(positions, tokens, predicate_1_1)
    attn_1_1_outputs = aggregate(attn_1_1_pattern, tokens)
    attn_1_1_output_scores = classifier_weights.loc[
        [("attn_1_1_outputs", str(v)) for v in attn_1_1_outputs]
    ]

    # attn_1_2 ####################################################
    def predicate_1_2(mlp_0_0_output, position):
        if mlp_0_0_output in {0, 3, 12, 5}:
            return position == 13
        elif mlp_0_0_output in {1}:
            return position == 1
        elif mlp_0_0_output in {2}:
            return position == 12
        elif mlp_0_0_output in {4}:
            return position == 6
        elif mlp_0_0_output in {6}:
            return position == 8
        elif mlp_0_0_output in {8, 11, 14, 7}:
            return position == 15
        elif mlp_0_0_output in {9, 13}:
            return position == 7
        elif mlp_0_0_output in {10}:
            return position == 10
        elif mlp_0_0_output in {15}:
            return position == 11

    attn_1_2_pattern = select_closest(positions, mlp_0_0_outputs, predicate_1_2)
    attn_1_2_outputs = aggregate(attn_1_2_pattern, tokens)
    attn_1_2_output_scores = classifier_weights.loc[
        [("attn_1_2_outputs", str(v)) for v in attn_1_2_outputs]
    ]

    # attn_1_3 ####################################################
    def predicate_1_3(token, position):
        if token in {"("}:
            return position == 1
        elif token in {")"}:
            return position == 10
        elif token in {"<s>"}:
            return position == 5

    attn_1_3_pattern = select_closest(positions, tokens, predicate_1_3)
    attn_1_3_outputs = aggregate(attn_1_3_pattern, tokens)
    attn_1_3_output_scores = classifier_weights.loc[
        [("attn_1_3_outputs", str(v)) for v in attn_1_3_outputs]
    ]

    # num_attn_1_0 ####################################################
    def num_predicate_1_0(num_mlp_0_0_output, position):
        if num_mlp_0_0_output in {0, 1, 8, 10, 12, 14}:
            return position == 5
        elif num_mlp_0_0_output in {9, 2, 4, 13}:
            return position == 4
        elif num_mlp_0_0_output in {3, 5, 6, 7, 11, 15}:
            return position == 3

    num_attn_1_0_pattern = select(positions, num_mlp_0_0_outputs, num_predicate_1_0)
    num_attn_1_0_outputs = aggregate_sum(num_attn_1_0_pattern, num_attn_0_0_outputs)
    num_attn_1_0_output_scores = classifier_weights.loc[
        [("num_attn_1_0_outputs", "_") for v in num_attn_1_0_outputs]
    ].mul(num_attn_1_0_outputs, axis=0)

    # num_attn_1_1 ####################################################
    def num_predicate_1_1(num_mlp_0_0_output, token):
        if num_mlp_0_0_output in {0, 1, 8, 9, 10, 12, 13}:
            return token == ""
        elif num_mlp_0_0_output in {2, 3, 4, 5, 6, 7, 11, 15}:
            return token == ")"
        elif num_mlp_0_0_output in {14}:
            return token == "<s>"

    num_attn_1_1_pattern = select(tokens, num_mlp_0_0_outputs, num_predicate_1_1)
    num_attn_1_1_outputs = aggregate_sum(num_attn_1_1_pattern, num_attn_0_0_outputs)
    num_attn_1_1_output_scores = classifier_weights.loc[
        [("num_attn_1_1_outputs", "_") for v in num_attn_1_1_outputs]
    ].mul(num_attn_1_1_outputs, axis=0)

    # num_attn_1_2 ####################################################
    def num_predicate_1_2(position, attn_0_2_output):
        if position in {0, 13, 14, 15}:
            return attn_0_2_output == ""
        elif position in {1, 3, 4, 5, 6, 7, 8}:
            return attn_0_2_output == ")"
        elif position in {2, 9, 10, 11, 12}:
            return attn_0_2_output == "<s>"

    num_attn_1_2_pattern = select(attn_0_2_outputs, positions, num_predicate_1_2)
    num_attn_1_2_outputs = aggregate_sum(num_attn_1_2_pattern, num_attn_0_0_outputs)
    num_attn_1_2_output_scores = classifier_weights.loc[
        [("num_attn_1_2_outputs", "_") for v in num_attn_1_2_outputs]
    ].mul(num_attn_1_2_outputs, axis=0)

    # num_attn_1_3 ####################################################
    def num_predicate_1_3(attn_0_0_output, token):
        if attn_0_0_output in {"("}:
            return token == ""
        elif attn_0_0_output in {"<s>", ")"}:
            return token == "("

    num_attn_1_3_pattern = select(tokens, attn_0_0_outputs, num_predicate_1_3)
    num_attn_1_3_outputs = aggregate_sum(num_attn_1_3_pattern, num_attn_0_1_outputs)
    num_attn_1_3_output_scores = classifier_weights.loc[
        [("num_attn_1_3_outputs", "_") for v in num_attn_1_3_outputs]
    ].mul(num_attn_1_3_outputs, axis=0)

    # mlp_1_0 #####################################################
    def mlp_1_0(attn_1_1_output, attn_0_3_output):
        key = (attn_1_1_output, attn_0_3_output)
        if key in {("(", "("), ("(", ")"), ("(", "<s>")}:
            return 13
        elif key in {(")", "("), ("<s>", "(")}:
            return 15
        return 4

    mlp_1_0_outputs = [
        mlp_1_0(k0, k1) for k0, k1 in zip(attn_1_1_outputs, attn_0_3_outputs)
    ]
    mlp_1_0_output_scores = classifier_weights.loc[
        [("mlp_1_0_outputs", str(v)) for v in mlp_1_0_outputs]
    ]

    # num_mlp_1_0 #################################################
    def num_mlp_1_0(num_attn_1_0_output):
        key = num_attn_1_0_output
        return 11

    num_mlp_1_0_outputs = [num_mlp_1_0(k0) for k0 in num_attn_1_0_outputs]
    num_mlp_1_0_output_scores = classifier_weights.loc[
        [("num_mlp_1_0_outputs", str(v)) for v in num_mlp_1_0_outputs]
    ]

    # attn_2_0 ####################################################
    def predicate_2_0(mlp_0_0_output, position):
        if mlp_0_0_output in {0, 5, 8, 11, 13}:
            return position == 7
        elif mlp_0_0_output in {1, 14}:
            return position == 3
        elif mlp_0_0_output in {9, 2, 12, 6}:
            return position == 11
        elif mlp_0_0_output in {3}:
            return position == 5
        elif mlp_0_0_output in {4}:
            return position == 10
        elif mlp_0_0_output in {10, 7}:
            return position == 13
        elif mlp_0_0_output in {15}:
            return position == 12

    attn_2_0_pattern = select_closest(positions, mlp_0_0_outputs, predicate_2_0)
    attn_2_0_outputs = aggregate(attn_2_0_pattern, attn_1_3_outputs)
    attn_2_0_output_scores = classifier_weights.loc[
        [("attn_2_0_outputs", str(v)) for v in attn_2_0_outputs]
    ]

    # attn_2_1 ####################################################
    def predicate_2_1(mlp_1_0_output, position):
        if mlp_1_0_output in {0, 2, 5, 7, 8, 11, 12}:
            return position == 5
        elif mlp_1_0_output in {1, 3, 9, 14}:
            return position == 1
        elif mlp_1_0_output in {4}:
            return position == 13
        elif mlp_1_0_output in {6}:
            return position == 15
        elif mlp_1_0_output in {10}:
            return position == 9
        elif mlp_1_0_output in {13}:
            return position == 12
        elif mlp_1_0_output in {15}:
            return position == 11

    attn_2_1_pattern = select_closest(positions, mlp_1_0_outputs, predicate_2_1)
    attn_2_1_outputs = aggregate(attn_2_1_pattern, attn_1_2_outputs)
    attn_2_1_output_scores = classifier_weights.loc[
        [("attn_2_1_outputs", str(v)) for v in attn_2_1_outputs]
    ]

    # attn_2_2 ####################################################
    def predicate_2_2(attn_0_1_output, position):
        if attn_0_1_output in {"(", ")"}:
            return position == 3
        elif attn_0_1_output in {"<s>"}:
            return position == 1

    attn_2_2_pattern = select_closest(positions, attn_0_1_outputs, predicate_2_2)
    attn_2_2_outputs = aggregate(attn_2_2_pattern, attn_1_0_outputs)
    attn_2_2_output_scores = classifier_weights.loc[
        [("attn_2_2_outputs", str(v)) for v in attn_2_2_outputs]
    ]

    # attn_2_3 ####################################################
    def predicate_2_3(mlp_1_0_output, position):
        if mlp_1_0_output in {0, 2, 5, 6, 7, 8, 12, 14}:
            return position == 5
        elif mlp_1_0_output in {1, 13}:
            return position == 2
        elif mlp_1_0_output in {11, 9, 10, 3}:
            return position == 3
        elif mlp_1_0_output in {4, 15}:
            return position == 1

    attn_2_3_pattern = select_closest(positions, mlp_1_0_outputs, predicate_2_3)
    attn_2_3_outputs = aggregate(attn_2_3_pattern, attn_0_3_outputs)
    attn_2_3_output_scores = classifier_weights.loc[
        [("attn_2_3_outputs", str(v)) for v in attn_2_3_outputs]
    ]

    # num_attn_2_0 ####################################################
    def num_predicate_2_0(attn_1_2_output, position):
        if attn_1_2_output in {"("}:
            return position == 6
        elif attn_1_2_output in {"<s>", ")"}:
            return position == 4

    num_attn_2_0_pattern = select(positions, attn_1_2_outputs, num_predicate_2_0)
    num_attn_2_0_outputs = aggregate_sum(num_attn_2_0_pattern, num_attn_1_2_outputs)
    num_attn_2_0_output_scores = classifier_weights.loc[
        [("num_attn_2_0_outputs", "_") for v in num_attn_2_0_outputs]
    ].mul(num_attn_2_0_outputs, axis=0)

    # num_attn_2_1 ####################################################
    def num_predicate_2_1(q_mlp_1_0_output, k_mlp_1_0_output):
        if q_mlp_1_0_output in {0, 2, 4, 5, 6, 8, 9, 11, 13, 14, 15}:
            return k_mlp_1_0_output == 15
        elif q_mlp_1_0_output in {1, 10, 3, 12}:
            return k_mlp_1_0_output == 2
        elif q_mlp_1_0_output in {7}:
            return k_mlp_1_0_output == 6

    num_attn_2_1_pattern = select(mlp_1_0_outputs, mlp_1_0_outputs, num_predicate_2_1)
    num_attn_2_1_outputs = aggregate_sum(num_attn_2_1_pattern, num_attn_0_2_outputs)
    num_attn_2_1_output_scores = classifier_weights.loc[
        [("num_attn_2_1_outputs", "_") for v in num_attn_2_1_outputs]
    ].mul(num_attn_2_1_outputs, axis=0)

    # num_attn_2_2 ####################################################
    def num_predicate_2_2(num_mlp_1_0_output, mlp_1_0_output):
        if num_mlp_1_0_output in {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}:
            return mlp_1_0_output == 4

    num_attn_2_2_pattern = select(
        mlp_1_0_outputs, num_mlp_1_0_outputs, num_predicate_2_2
    )
    num_attn_2_2_outputs = aggregate_sum(num_attn_2_2_pattern, num_attn_0_2_outputs)
    num_attn_2_2_output_scores = classifier_weights.loc[
        [("num_attn_2_2_outputs", "_") for v in num_attn_2_2_outputs]
    ].mul(num_attn_2_2_outputs, axis=0)

    # num_attn_2_3 ####################################################
    def num_predicate_2_3(attn_1_0_output, mlp_0_0_output):
        if attn_1_0_output in {"<s>", "(", ")"}:
            return mlp_0_0_output == 2

    num_attn_2_3_pattern = select(mlp_0_0_outputs, attn_1_0_outputs, num_predicate_2_3)
    num_attn_2_3_outputs = aggregate_sum(num_attn_2_3_pattern, num_attn_0_2_outputs)
    num_attn_2_3_output_scores = classifier_weights.loc[
        [("num_attn_2_3_outputs", "_") for v in num_attn_2_3_outputs]
    ].mul(num_attn_2_3_outputs, axis=0)

    # mlp_2_0 #####################################################
    def mlp_2_0(attn_2_3_output, attn_2_2_output):
        key = (attn_2_3_output, attn_2_2_output)
        if key in {("(", ")"), (")", "("), ("<s>", "(")}:
            return 6
        elif key in {("(", "("), ("(", "<s>")}:
            return 3
        return 9

    mlp_2_0_outputs = [
        mlp_2_0(k0, k1) for k0, k1 in zip(attn_2_3_outputs, attn_2_2_outputs)
    ]
    mlp_2_0_output_scores = classifier_weights.loc[
        [("mlp_2_0_outputs", str(v)) for v in mlp_2_0_outputs]
    ]

    # num_mlp_2_0 #################################################
    def num_mlp_2_0(num_attn_2_1_output, num_attn_0_0_output):
        key = (num_attn_2_1_output, num_attn_0_0_output)
        if key in {
            (0, 0),
            (0, 1),
            (0, 2),
            (0, 3),
            (0, 4),
            (0, 5),
            (0, 6),
            (0, 7),
            (0, 8),
            (0, 9),
            (0, 10),
            (0, 11),
            (0, 12),
            (0, 13),
            (0, 14),
            (0, 15),
            (0, 16),
            (0, 17),
            (0, 18),
            (0, 19),
            (0, 20),
            (0, 21),
            (0, 22),
            (0, 23),
            (0, 24),
            (0, 25),
            (0, 26),
            (0, 27),
            (0, 28),
            (0, 29),
            (0, 30),
            (0, 31),
            (0, 32),
            (0, 33),
            (0, 34),
            (0, 35),
            (0, 36),
            (0, 37),
            (0, 38),
            (0, 39),
            (0, 40),
            (0, 41),
            (0, 42),
            (0, 43),
            (0, 44),
            (0, 45),
            (0, 46),
            (0, 47),
            (1, 11),
            (1, 12),
            (1, 13),
            (1, 14),
            (1, 15),
            (1, 16),
            (1, 17),
            (1, 18),
            (1, 19),
            (1, 20),
            (1, 21),
            (1, 22),
            (1, 23),
            (1, 24),
            (1, 25),
            (1, 26),
            (1, 27),
            (1, 28),
            (1, 29),
            (1, 30),
            (1, 31),
            (1, 32),
            (1, 33),
            (1, 34),
            (1, 35),
            (1, 36),
            (1, 37),
            (1, 38),
            (1, 39),
            (1, 40),
            (1, 41),
            (1, 42),
            (1, 43),
            (1, 44),
            (1, 45),
            (1, 46),
            (1, 47),
            (2, 24),
            (2, 25),
            (2, 26),
            (2, 27),
            (2, 28),
            (2, 29),
            (2, 30),
            (2, 31),
            (2, 32),
            (2, 33),
            (2, 34),
            (2, 35),
            (2, 36),
            (2, 37),
            (2, 38),
            (2, 39),
            (2, 40),
            (2, 41),
            (2, 42),
            (2, 43),
            (2, 44),
            (2, 45),
            (2, 46),
            (2, 47),
            (3, 36),
            (3, 37),
            (3, 38),
            (3, 39),
            (3, 40),
            (3, 41),
            (3, 42),
            (3, 43),
            (3, 44),
            (3, 45),
            (3, 46),
            (3, 47),
        }:
            return 8
        return 12

    num_mlp_2_0_outputs = [
        num_mlp_2_0(k0, k1)
        for k0, k1 in zip(num_attn_2_1_outputs, num_attn_0_0_outputs)
    ]
    num_mlp_2_0_output_scores = classifier_weights.loc[
        [("num_mlp_2_0_outputs", str(v)) for v in num_mlp_2_0_outputs]
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
                mlp_0_0_output_scores,
                num_mlp_0_0_output_scores,
                attn_1_0_output_scores,
                attn_1_1_output_scores,
                attn_1_2_output_scores,
                attn_1_3_output_scores,
                mlp_1_0_output_scores,
                num_mlp_1_0_output_scores,
                attn_2_0_output_scores,
                attn_2_1_output_scores,
                attn_2_2_output_scores,
                attn_2_3_output_scores,
                mlp_2_0_output_scores,
                num_mlp_2_0_output_scores,
                one_scores,
                num_attn_0_0_output_scores,
                num_attn_0_1_output_scores,
                num_attn_0_2_output_scores,
                num_attn_0_3_output_scores,
                num_attn_1_0_output_scores,
                num_attn_1_1_output_scores,
                num_attn_1_2_output_scores,
                num_attn_1_3_output_scores,
                num_attn_2_0_output_scores,
                num_attn_2_1_output_scores,
                num_attn_2_2_output_scores,
                num_attn_2_3_output_scores,
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
            "(",
            ")",
            "(",
            "(",
            ")",
            ")",
            "(",
            "(",
            ")",
            ")",
            "(",
            ")",
            ")",
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
            "(",
            "(",
            ")",
            ")",
            ")",
            ")",
            "(",
            "(",
            "(",
            "(",
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
            ")",
            ")",
            "(",
            "(",
            "(",
            "(",
            "(",
            ")",
            ")",
            ")",
            "(",
            "(",
            "(",
            ")",
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
            "(",
            "(",
            "(",
            "(",
            ")",
            "(",
            ")",
            "(",
            "(",
            "(",
            "(",
            "(",
            "(",
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
            ")",
            "(",
            "(",
            ")",
            "(",
            ")",
            "(",
            ")",
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
            "(",
            ")",
            ")",
            "(",
            ")",
            ")",
            ")",
            "(",
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
            "(",
            ")",
            ")",
            "(",
            ")",
            "(",
            ")",
            ")",
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
            "(",
            ")",
            ")",
            "(",
            "(",
            ")",
            "(",
            "(",
            "(",
            "(",
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
            "(",
            "(",
            "(",
            "(",
            "(",
            ")",
            ")",
            "(",
            ")",
            ")",
            "(",
            ")",
            ")",
            ")",
            "(",
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
            "T",
            "P",
        ],
    ),
    (
        [
            "<s>",
            "(",
            ")",
            ")",
            ")",
            ")",
            ")",
            "(",
            "(",
            ")",
            "(",
            ")",
            ")",
            ")",
            "(",
            ")",
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
]
for x, y in examples:
    print(f"x: {x}")
    print(f"y: {y}")
    y_hat = run(x)
    print(f"y_hat: {y_hat}")
    print()
