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
        "programs/rasp_categorical_only/dyck2/dyck2_weights.csv",
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
        if q_position in {0}:
            return k_position == 14
        elif q_position in {1}:
            return k_position == 4
        elif q_position in {2}:
            return k_position == 15
        elif q_position in {3}:
            return k_position == 5
        elif q_position in {4}:
            return k_position == 1
        elif q_position in {5, 15}:
            return k_position == 2
        elif q_position in {6}:
            return k_position == 3
        elif q_position in {8, 7}:
            return k_position == 6
        elif q_position in {9}:
            return k_position == 8
        elif q_position in {10}:
            return k_position == 9
        elif q_position in {11}:
            return k_position == 10
        elif q_position in {12}:
            return k_position == 11
        elif q_position in {13, 14}:
            return k_position == 12

    attn_0_0_pattern = select_closest(positions, positions, predicate_0_0)
    attn_0_0_outputs = aggregate(attn_0_0_pattern, tokens)
    attn_0_0_output_scores = classifier_weights.loc[
        [("attn_0_0_outputs", str(v)) for v in attn_0_0_outputs]
    ]

    # attn_0_1 ####################################################
    def predicate_0_1(q_position, k_position):
        if q_position in {0}:
            return k_position == 15
        elif q_position in {1}:
            return k_position == 0
        elif q_position in {2}:
            return k_position == 1
        elif q_position in {3}:
            return k_position == 2
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

    attn_0_1_pattern = select_closest(positions, positions, predicate_0_1)
    attn_0_1_outputs = aggregate(attn_0_1_pattern, tokens)
    attn_0_1_output_scores = classifier_weights.loc[
        [("attn_0_1_outputs", str(v)) for v in attn_0_1_outputs]
    ]

    # attn_0_2 ####################################################
    def predicate_0_2(token, position):
        if token in {"{", "}", "(", ")"}:
            return position == 0
        elif token in {"<s>"}:
            return position == 10

    attn_0_2_pattern = select_closest(positions, tokens, predicate_0_2)
    attn_0_2_outputs = aggregate(attn_0_2_pattern, tokens)
    attn_0_2_output_scores = classifier_weights.loc[
        [("attn_0_2_outputs", str(v)) for v in attn_0_2_outputs]
    ]

    # attn_0_3 ####################################################
    def predicate_0_3(q_position, k_position):
        if q_position in {0, 2, 14}:
            return k_position == 13
        elif q_position in {1}:
            return k_position == 0
        elif q_position in {3, 12}:
            return k_position == 11
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
        elif q_position in {13}:
            return k_position == 12
        elif q_position in {15}:
            return k_position == 14

    attn_0_3_pattern = select_closest(positions, positions, predicate_0_3)
    attn_0_3_outputs = aggregate(attn_0_3_pattern, tokens)
    attn_0_3_output_scores = classifier_weights.loc[
        [("attn_0_3_outputs", str(v)) for v in attn_0_3_outputs]
    ]

    # mlp_0_0 #####################################################
    def mlp_0_0(attn_0_1_output, token):
        key = (attn_0_1_output, token)
        if key in {(")", ")"), (")", "}"), ("}", ")"), ("}", "}")}:
            return 2
        elif key in {("(", "}"), ("<s>", ")"), ("<s>", "}"), ("{", ")")}:
            return 5
        elif key in {
            ("(", "("),
            ("(", "<s>"),
            ("(", "{"),
            ("<s>", "("),
            ("<s>", "<s>"),
            ("<s>", "{"),
            ("{", "("),
            ("{", "<s>"),
            ("{", "{"),
        }:
            return 12
        elif key in {("(", ")"), ("{", "}")}:
            return 10
        elif key in {(")", "("), (")", "<s>"), (")", "{")}:
            return 9
        return 8

    mlp_0_0_outputs = [mlp_0_0(k0, k1) for k0, k1 in zip(attn_0_1_outputs, tokens)]
    mlp_0_0_output_scores = classifier_weights.loc[
        [("mlp_0_0_outputs", str(v)) for v in mlp_0_0_outputs]
    ]

    # mlp_0_1 #####################################################
    def mlp_0_1(token, attn_0_0_output):
        key = (token, attn_0_0_output)
        if key in {
            (")", ")"),
            (")", "<s>"),
            (")", "}"),
            ("<s>", ")"),
            ("<s>", "<s>"),
            ("<s>", "}"),
            ("}", ")"),
            ("}", "<s>"),
            ("}", "}"),
        }:
            return 5
        elif key in {(")", "(")}:
            return 9
        elif key in {("(", ")"), ("(", "}"), ("{", ")"), ("{", "}")}:
            return 13
        elif key in {
            ("(", "("),
            ("(", "<s>"),
            ("(", "{"),
            ("<s>", "("),
            ("<s>", "{"),
            ("{", "("),
            ("{", "<s>"),
            ("{", "{"),
        }:
            return 12
        elif key in {("}", "{")}:
            return 7
        return 15

    mlp_0_1_outputs = [mlp_0_1(k0, k1) for k0, k1 in zip(tokens, attn_0_0_outputs)]
    mlp_0_1_output_scores = classifier_weights.loc[
        [("mlp_0_1_outputs", str(v)) for v in mlp_0_1_outputs]
    ]

    # attn_1_0 ####################################################
    def predicate_1_0(q_position, k_position):
        if q_position in {0}:
            return k_position == 15
        elif q_position in {1}:
            return k_position == 1
        elif q_position in {2, 5, 6}:
            return k_position == 2
        elif q_position in {3, 7}:
            return k_position == 3
        elif q_position in {4}:
            return k_position == 5
        elif q_position in {8, 9, 11, 13}:
            return k_position == 7
        elif q_position in {10, 12}:
            return k_position == 8
        elif q_position in {14}:
            return k_position == 11
        elif q_position in {15}:
            return k_position == 13

    attn_1_0_pattern = select_closest(positions, positions, predicate_1_0)
    attn_1_0_outputs = aggregate(attn_1_0_pattern, mlp_0_0_outputs)
    attn_1_0_output_scores = classifier_weights.loc[
        [("attn_1_0_outputs", str(v)) for v in attn_1_0_outputs]
    ]

    # attn_1_1 ####################################################
    def predicate_1_1(q_position, k_position):
        if q_position in {0, 11}:
            return k_position == 4
        elif q_position in {1}:
            return k_position == 1
        elif q_position in {9, 2}:
            return k_position == 2
        elif q_position in {3, 4, 5, 7}:
            return k_position == 3
        elif q_position in {8, 13, 6}:
            return k_position == 5
        elif q_position in {10, 12}:
            return k_position == 6
        elif q_position in {14, 15}:
            return k_position == 7

    attn_1_1_pattern = select_closest(positions, positions, predicate_1_1)
    attn_1_1_outputs = aggregate(attn_1_1_pattern, mlp_0_0_outputs)
    attn_1_1_output_scores = classifier_weights.loc[
        [("attn_1_1_outputs", str(v)) for v in attn_1_1_outputs]
    ]

    # attn_1_2 ####################################################
    def predicate_1_2(token, position):
        if token in {"{", "("}:
            return position == 3
        elif token in {"}", ")"}:
            return position == 5
        elif token in {"<s>"}:
            return position == 13

    attn_1_2_pattern = select_closest(positions, tokens, predicate_1_2)
    attn_1_2_outputs = aggregate(attn_1_2_pattern, mlp_0_0_outputs)
    attn_1_2_output_scores = classifier_weights.loc[
        [("attn_1_2_outputs", str(v)) for v in attn_1_2_outputs]
    ]

    # attn_1_3 ####################################################
    def predicate_1_3(q_position, k_position):
        if q_position in {0}:
            return k_position == 15
        elif q_position in {1}:
            return k_position == 1
        elif q_position in {2, 4}:
            return k_position == 2
        elif q_position in {9, 3}:
            return k_position == 3
        elif q_position in {5, 7}:
            return k_position == 4
        elif q_position in {6}:
            return k_position == 5
        elif q_position in {8}:
            return k_position == 6
        elif q_position in {10}:
            return k_position == 7
        elif q_position in {11, 13, 14, 15}:
            return k_position == 9
        elif q_position in {12}:
            return k_position == 10

    attn_1_3_pattern = select_closest(positions, positions, predicate_1_3)
    attn_1_3_outputs = aggregate(attn_1_3_pattern, mlp_0_1_outputs)
    attn_1_3_output_scores = classifier_weights.loc[
        [("attn_1_3_outputs", str(v)) for v in attn_1_3_outputs]
    ]

    # mlp_1_0 #####################################################
    def mlp_1_0(attn_1_1_output, attn_1_3_output):
        key = (attn_1_1_output, attn_1_3_output)
        if key in {
            (0, 4),
            (0, 6),
            (0, 14),
            (1, 0),
            (1, 2),
            (1, 5),
            (1, 14),
            (1, 15),
            (2, 14),
            (3, 2),
            (3, 14),
            (4, 0),
            (4, 2),
            (4, 3),
            (4, 5),
            (4, 10),
            (4, 14),
            (5, 1),
            (5, 4),
            (5, 6),
            (5, 11),
            (5, 14),
            (6, 4),
            (6, 6),
            (6, 8),
            (6, 11),
            (6, 14),
            (7, 14),
            (9, 0),
            (9, 2),
            (9, 3),
            (9, 5),
            (9, 14),
            (11, 2),
            (11, 5),
            (11, 14),
            (11, 15),
            (12, 0),
            (12, 1),
            (12, 2),
            (12, 3),
            (12, 4),
            (12, 5),
            (12, 6),
            (12, 7),
            (12, 8),
            (12, 9),
            (12, 10),
            (12, 11),
            (12, 12),
            (12, 13),
            (12, 14),
            (12, 15),
            (13, 14),
            (14, 2),
            (14, 14),
            (15, 1),
            (15, 2),
            (15, 4),
            (15, 6),
            (15, 8),
            (15, 14),
        }:
            return 3
        elif key in {
            (0, 11),
            (1, 1),
            (1, 4),
            (1, 6),
            (1, 8),
            (1, 9),
            (1, 10),
            (1, 11),
            (1, 12),
            (4, 1),
            (4, 4),
            (4, 6),
            (4, 8),
            (4, 9),
            (4, 11),
            (4, 12),
            (8, 9),
            (9, 1),
            (9, 4),
            (9, 6),
            (9, 8),
            (9, 9),
            (9, 11),
            (9, 12),
            (11, 0),
            (11, 1),
            (11, 4),
            (11, 6),
            (11, 8),
            (11, 9),
            (11, 10),
            (11, 11),
            (11, 12),
            (14, 1),
            (14, 4),
            (14, 6),
            (14, 8),
            (14, 11),
            (14, 12),
            (15, 11),
            (15, 12),
        }:
            return 10
        elif key in {
            (0, 8),
            (0, 12),
            (5, 12),
            (6, 12),
            (7, 12),
            (8, 0),
            (8, 1),
            (8, 2),
            (8, 3),
            (8, 4),
            (8, 5),
            (8, 6),
            (8, 8),
            (8, 10),
            (8, 11),
            (8, 12),
            (8, 13),
            (8, 14),
            (9, 10),
            (9, 13),
            (10, 1),
            (10, 4),
            (10, 12),
            (11, 13),
            (13, 12),
        }:
            return 11
        elif key in {
            (0, 1),
            (1, 3),
            (2, 0),
            (2, 1),
            (2, 3),
            (2, 4),
            (2, 6),
            (2, 11),
            (2, 12),
            (3, 0),
            (3, 1),
            (3, 3),
            (3, 4),
            (3, 6),
            (3, 10),
            (3, 11),
            (3, 12),
            (6, 1),
            (7, 1),
            (11, 3),
            (13, 1),
            (13, 3),
            (13, 4),
            (13, 6),
        }:
            return 13
        elif key in {
            (0, 15),
            (3, 15),
            (4, 15),
            (5, 15),
            (6, 15),
            (8, 15),
            (9, 15),
            (14, 15),
            (15, 15),
        }:
            return 7
        return 14

    mlp_1_0_outputs = [
        mlp_1_0(k0, k1) for k0, k1 in zip(attn_1_1_outputs, attn_1_3_outputs)
    ]
    mlp_1_0_output_scores = classifier_weights.loc[
        [("mlp_1_0_outputs", str(v)) for v in mlp_1_0_outputs]
    ]

    # mlp_1_1 #####################################################
    def mlp_1_1(mlp_0_0_output, attn_1_1_output):
        key = (mlp_0_0_output, attn_1_1_output)
        if key in {
            (0, 1),
            (0, 4),
            (0, 11),
            (0, 12),
            (0, 15),
            (1, 0),
            (1, 1),
            (1, 2),
            (1, 3),
            (1, 4),
            (1, 5),
            (1, 6),
            (1, 7),
            (1, 8),
            (1, 9),
            (1, 11),
            (1, 12),
            (1, 13),
            (1, 14),
            (1, 15),
            (3, 1),
            (3, 11),
            (3, 12),
            (3, 15),
            (4, 0),
            (4, 1),
            (4, 4),
            (4, 5),
            (4, 6),
            (4, 7),
            (4, 8),
            (4, 11),
            (4, 12),
            (4, 13),
            (4, 14),
            (4, 15),
            (5, 11),
            (5, 12),
            (6, 1),
            (6, 11),
            (6, 12),
            (6, 15),
            (7, 1),
            (7, 4),
            (7, 11),
            (7, 12),
            (7, 13),
            (7, 15),
            (8, 1),
            (8, 4),
            (8, 11),
            (8, 12),
            (8, 14),
            (8, 15),
            (9, 11),
            (9, 12),
            (9, 15),
            (10, 1),
            (10, 4),
            (10, 11),
            (10, 12),
            (10, 15),
            (11, 0),
            (11, 1),
            (11, 2),
            (11, 3),
            (11, 4),
            (11, 5),
            (11, 6),
            (11, 7),
            (11, 8),
            (11, 9),
            (11, 10),
            (11, 11),
            (11, 12),
            (11, 13),
            (11, 14),
            (11, 15),
            (12, 0),
            (12, 1),
            (12, 2),
            (12, 3),
            (12, 4),
            (12, 5),
            (12, 6),
            (12, 7),
            (12, 8),
            (12, 9),
            (12, 10),
            (12, 11),
            (12, 12),
            (12, 13),
            (12, 14),
            (12, 15),
            (13, 1),
            (13, 4),
            (13, 11),
            (13, 12),
            (13, 15),
            (14, 1),
            (14, 11),
            (14, 12),
            (14, 15),
            (15, 0),
            (15, 1),
            (15, 2),
            (15, 3),
            (15, 4),
            (15, 5),
            (15, 6),
            (15, 7),
            (15, 8),
            (15, 9),
            (15, 10),
            (15, 11),
            (15, 12),
            (15, 13),
            (15, 14),
            (15, 15),
        }:
            return 1
        elif key in {
            (6, 8),
            (7, 8),
            (10, 0),
            (10, 2),
            (10, 3),
            (10, 5),
            (10, 6),
            (10, 7),
            (10, 8),
            (10, 9),
            (10, 10),
            (10, 13),
            (10, 14),
            (14, 8),
        }:
            return 9
        return 14

    mlp_1_1_outputs = [
        mlp_1_1(k0, k1) for k0, k1 in zip(mlp_0_0_outputs, attn_1_1_outputs)
    ]
    mlp_1_1_output_scores = classifier_weights.loc[
        [("mlp_1_1_outputs", str(v)) for v in mlp_1_1_outputs]
    ]

    # attn_2_0 ####################################################
    def predicate_2_0(token, position):
        if token in {"{", "("}:
            return position == 5
        elif token in {"<s>", "}", ")"}:
            return position == 4

    attn_2_0_pattern = select_closest(positions, tokens, predicate_2_0)
    attn_2_0_outputs = aggregate(attn_2_0_pattern, mlp_0_0_outputs)
    attn_2_0_output_scores = classifier_weights.loc[
        [("attn_2_0_outputs", str(v)) for v in attn_2_0_outputs]
    ]

    # attn_2_1 ####################################################
    def predicate_2_1(mlp_1_1_output, mlp_0_0_output):
        if mlp_1_1_output in {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14}:
            return mlp_0_0_output == 5
        elif mlp_1_1_output in {15}:
            return mlp_0_0_output == 1

    attn_2_1_pattern = select_closest(mlp_0_0_outputs, mlp_1_1_outputs, predicate_2_1)
    attn_2_1_outputs = aggregate(attn_2_1_pattern, mlp_0_0_outputs)
    attn_2_1_output_scores = classifier_weights.loc[
        [("attn_2_1_outputs", str(v)) for v in attn_2_1_outputs]
    ]

    # attn_2_2 ####################################################
    def predicate_2_2(attn_0_2_output, mlp_0_0_output):
        if attn_0_2_output in {"}", "(", ")"}:
            return mlp_0_0_output == 5
        elif attn_0_2_output in {"<s>", "{"}:
            return mlp_0_0_output == 2

    attn_2_2_pattern = select_closest(mlp_0_0_outputs, attn_0_2_outputs, predicate_2_2)
    attn_2_2_outputs = aggregate(attn_2_2_pattern, mlp_0_0_outputs)
    attn_2_2_output_scores = classifier_weights.loc[
        [("attn_2_2_outputs", str(v)) for v in attn_2_2_outputs]
    ]

    # attn_2_3 ####################################################
    def predicate_2_3(q_position, k_position):
        if q_position in {0}:
            return k_position == 4
        elif q_position in {1}:
            return k_position == 1
        elif q_position in {2, 3}:
            return k_position == 2
        elif q_position in {4, 5, 6, 7, 8, 9, 10, 11, 13, 15}:
            return k_position == 3
        elif q_position in {12, 14}:
            return k_position == 5

    attn_2_3_pattern = select_closest(positions, positions, predicate_2_3)
    attn_2_3_outputs = aggregate(attn_2_3_pattern, attn_1_1_outputs)
    attn_2_3_output_scores = classifier_weights.loc[
        [("attn_2_3_outputs", str(v)) for v in attn_2_3_outputs]
    ]

    # mlp_2_0 #####################################################
    def mlp_2_0(attn_2_3_output, attn_2_2_output):
        key = (attn_2_3_output, attn_2_2_output)
        if key in {
            (0, 2),
            (0, 3),
            (0, 5),
            (0, 7),
            (0, 14),
            (2, 0),
            (2, 2),
            (2, 3),
            (2, 5),
            (2, 6),
            (2, 7),
            (2, 8),
            (2, 9),
            (2, 10),
            (2, 11),
            (2, 13),
            (2, 14),
            (2, 15),
            (3, 0),
            (3, 2),
            (3, 3),
            (3, 5),
            (3, 6),
            (3, 7),
            (3, 8),
            (3, 9),
            (3, 10),
            (3, 11),
            (3, 13),
            (3, 14),
            (3, 15),
            (5, 2),
            (5, 3),
            (5, 5),
            (5, 7),
            (5, 14),
            (6, 2),
            (6, 3),
            (6, 5),
            (7, 0),
            (7, 2),
            (7, 3),
            (7, 5),
            (7, 7),
            (7, 11),
            (7, 14),
            (7, 15),
            (8, 2),
            (8, 3),
            (8, 5),
            (8, 7),
            (8, 14),
            (9, 0),
            (9, 2),
            (9, 3),
            (9, 5),
            (9, 7),
            (9, 10),
            (9, 11),
            (9, 14),
            (11, 0),
            (11, 2),
            (11, 3),
            (11, 5),
            (11, 7),
            (11, 11),
            (11, 14),
            (13, 0),
            (13, 2),
            (13, 3),
            (13, 5),
            (13, 6),
            (13, 7),
            (13, 9),
            (13, 11),
            (13, 13),
            (13, 14),
            (13, 15),
            (14, 0),
            (14, 2),
            (14, 3),
            (14, 5),
            (14, 7),
            (14, 11),
            (14, 14),
            (15, 0),
            (15, 2),
            (15, 3),
            (15, 5),
            (15, 7),
            (15, 14),
        }:
            return 6
        elif key in {
            (6, 13),
            (6, 15),
            (7, 12),
            (8, 0),
            (8, 1),
            (8, 4),
            (8, 6),
            (8, 8),
            (8, 9),
            (8, 10),
            (8, 11),
            (8, 12),
            (8, 13),
            (8, 15),
            (9, 1),
            (9, 4),
            (9, 6),
            (9, 8),
            (9, 9),
            (9, 12),
            (9, 13),
            (9, 15),
            (11, 12),
            (11, 13),
            (11, 15),
            (13, 12),
            (14, 12),
            (14, 13),
            (14, 15),
            (15, 6),
            (15, 8),
            (15, 9),
            (15, 11),
            (15, 12),
            (15, 13),
            (15, 15),
        }:
            return 7
        elif key in {
            (1, 1),
            (1, 4),
            (1, 6),
            (1, 8),
            (1, 9),
            (1, 11),
            (1, 12),
            (1, 13),
            (1, 14),
            (1, 15),
            (4, 1),
            (4, 4),
            (4, 6),
            (4, 8),
            (4, 11),
            (4, 12),
            (4, 13),
            (4, 14),
            (4, 15),
            (6, 1),
            (6, 12),
            (12, 1),
            (12, 12),
        }:
            return 0
        return 9

    mlp_2_0_outputs = [
        mlp_2_0(k0, k1) for k0, k1 in zip(attn_2_3_outputs, attn_2_2_outputs)
    ]
    mlp_2_0_output_scores = classifier_weights.loc[
        [("mlp_2_0_outputs", str(v)) for v in mlp_2_0_outputs]
    ]

    # mlp_2_1 #####################################################
    def mlp_2_1(attn_2_2_output, attn_2_1_output):
        key = (attn_2_2_output, attn_2_1_output)
        if key in {
            (0, 5),
            (0, 15),
            (1, 5),
            (1, 15),
            (2, 5),
            (2, 15),
            (3, 5),
            (3, 15),
            (4, 5),
            (5, 0),
            (5, 1),
            (5, 2),
            (5, 3),
            (5, 4),
            (5, 5),
            (5, 6),
            (5, 7),
            (5, 8),
            (5, 9),
            (5, 10),
            (5, 11),
            (5, 12),
            (5, 13),
            (5, 14),
            (5, 15),
            (6, 5),
            (6, 15),
            (7, 0),
            (7, 5),
            (7, 6),
            (7, 7),
            (7, 11),
            (7, 14),
            (7, 15),
            (8, 5),
            (8, 15),
            (9, 5),
            (9, 15),
            (10, 5),
            (10, 15),
            (11, 0),
            (11, 1),
            (11, 3),
            (11, 5),
            (11, 6),
            (11, 7),
            (11, 10),
            (11, 11),
            (11, 14),
            (11, 15),
            (12, 5),
            (12, 15),
            (13, 5),
            (13, 15),
            (14, 5),
            (14, 15),
            (15, 0),
            (15, 2),
            (15, 3),
            (15, 5),
            (15, 6),
            (15, 7),
            (15, 10),
            (15, 11),
            (15, 14),
            (15, 15),
        }:
            return 0
        elif key in {(4, 4), (4, 15), (6, 4), (7, 4), (13, 4), (15, 4)}:
            return 2
        return 10

    mlp_2_1_outputs = [
        mlp_2_1(k0, k1) for k0, k1 in zip(attn_2_2_outputs, attn_2_1_outputs)
    ]
    mlp_2_1_output_scores = classifier_weights.loc[
        [("mlp_2_1_outputs", str(v)) for v in mlp_2_1_outputs]
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
                mlp_0_1_output_scores,
                attn_1_0_output_scores,
                attn_1_1_output_scores,
                attn_1_2_output_scores,
                attn_1_3_output_scores,
                mlp_1_0_output_scores,
                mlp_1_1_output_scores,
                attn_2_0_output_scores,
                attn_2_1_output_scores,
                attn_2_2_output_scores,
                attn_2_3_output_scores,
                mlp_2_0_output_scores,
                mlp_2_1_output_scores,
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
            "{",
            "{",
            "{",
            "(",
            ")",
            "(",
            ")",
            ")",
            "(",
            "}",
            "{",
            ")",
            "{",
            "}",
            "}",
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
            "}",
            "{",
            "}",
            "}",
            ")",
            "(",
            ")",
            "(",
            "(",
            "{",
            ")",
            "}",
            ")",
            "}",
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
            "{",
            "{",
            "(",
            "{",
            "{",
            "(",
            ")",
            "}",
            "(",
            ")",
            "}",
            ")",
            "(",
            ")",
            "}",
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
            "}",
            "{",
            "(",
            ")",
            "{",
            ")",
            ")",
            "(",
            ")",
            "}",
            ")",
            ")",
            "{",
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
            "{",
            "(",
            "{",
            "{",
            "}",
            "}",
            "{",
            "}",
            "(",
            ")",
            ")",
            "}",
            ")",
            "{",
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
            "(",
            "(",
            "{",
            "(",
            ")",
            "}",
            ")",
            ")",
            "(",
            ")",
            ")",
            "(",
            ")",
            "{",
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
            "T",
            "P",
            "T",
            "P",
        ],
    ),
    (
        [
            "<s>",
            "(",
            "(",
            "}",
            "{",
            ")",
            "(",
            "}",
            "}",
            ")",
            "(",
            "(",
            "{",
            "}",
            "{",
            "(",
        ],
        [
            "<pad>",
            "P",
            "P",
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
            "{",
            "{",
            "{",
            "{",
            "(",
            "{",
            "}",
            ")",
            "}",
            "}",
            "}",
            "}",
            ")",
            "{",
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
            "{",
            "{",
            "{",
            ")",
            ")",
            "(",
            "{",
            "}",
            "{",
            "}",
            "}",
            ")",
            ")",
            ")",
            ")",
        ],
        [
            "<pad>",
            "P",
            "P",
            "P",
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
            "{",
            "{",
            ")",
            "{",
            "{",
            "{",
            "(",
            "}",
            "}",
            "(",
            ")",
            ")",
            "(",
            ")",
            "{",
        ],
        [
            "<pad>",
            "P",
            "P",
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
