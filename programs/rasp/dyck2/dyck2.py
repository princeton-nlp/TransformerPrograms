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
        "programs/rasp/dyck2/dyck2_weights.csv",
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
    def predicate_0_0(token, position):
        if token in {"{", "("}:
            return position == 15
        elif token in {"}", ")"}:
            return position == 4
        elif token in {"<s>"}:
            return position == 8

    attn_0_0_pattern = select_closest(positions, tokens, predicate_0_0)
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

    # num_attn_0_0 ####################################################
    def num_predicate_0_0(q_token, k_token):
        if q_token in {"{", "("}:
            return k_token == "<pad>"
        elif q_token in {")"}:
            return k_token == "("
        elif q_token in {"<s>", "}"}:
            return k_token == "{"

    num_attn_0_0_pattern = select(tokens, tokens, num_predicate_0_0)
    num_attn_0_0_outputs = aggregate_sum(num_attn_0_0_pattern, ones)
    num_attn_0_0_output_scores = classifier_weights.loc[
        [("num_attn_0_0_outputs", "_") for v in num_attn_0_0_outputs]
    ].mul(num_attn_0_0_outputs, axis=0)

    # num_attn_0_1 ####################################################
    def num_predicate_0_1(position, token):
        if position in {0, 1}:
            return token == "}"
        elif position in {2, 3}:
            return token == "("
        elif position in {4, 6, 7, 8, 14}:
            return token == ""
        elif position in {5}:
            return token == "{"
        elif position in {9, 10}:
            return token == "<s>"
        elif position in {11, 13, 15}:
            return token == ")"
        elif position in {12}:
            return token == "<pad>"

    num_attn_0_1_pattern = select(tokens, positions, num_predicate_0_1)
    num_attn_0_1_outputs = aggregate_sum(num_attn_0_1_pattern, ones)
    num_attn_0_1_output_scores = classifier_weights.loc[
        [("num_attn_0_1_outputs", "_") for v in num_attn_0_1_outputs]
    ].mul(num_attn_0_1_outputs, axis=0)

    # mlp_0_0 #####################################################
    def mlp_0_0(attn_0_1_output, token):
        key = (attn_0_1_output, token)
        if key in {("(", "}"), ("<s>", "}"), ("{", ")")}:
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
            return 4
        elif key in {(")", "}"), ("}", ")"), ("}", "<s>"), ("}", "}")}:
            return 6
        return 7

    mlp_0_0_outputs = [mlp_0_0(k0, k1) for k0, k1 in zip(attn_0_1_outputs, tokens)]
    mlp_0_0_output_scores = classifier_weights.loc[
        [("mlp_0_0_outputs", str(v)) for v in mlp_0_0_outputs]
    ]

    # mlp_0_1 #####################################################
    def mlp_0_1(attn_0_1_output, token):
        key = (attn_0_1_output, token)
        if key in {(")", "("), (")", "{"), ("}", "("), ("}", "{")}:
            return 8
        elif key in {
            (")", ")"),
            (")", "<s>"),
            (")", "}"),
            ("<s>", ")"),
            ("<s>", "}"),
            ("}", ")"),
            ("}", "<s>"),
            ("}", "}"),
        }:
            return 5
        elif key in {
            ("(", "<s>"),
            ("<s>", "("),
            ("<s>", "<s>"),
            ("<s>", "{"),
            ("{", "<s>"),
        }:
            return 4
        elif key in {("(", "("), ("(", "{"), ("{", "("), ("{", "{")}:
            return 1
        return 14

    mlp_0_1_outputs = [mlp_0_1(k0, k1) for k0, k1 in zip(attn_0_1_outputs, tokens)]
    mlp_0_1_output_scores = classifier_weights.loc[
        [("mlp_0_1_outputs", str(v)) for v in mlp_0_1_outputs]
    ]

    # num_mlp_0_0 #################################################
    def num_mlp_0_0(num_attn_0_1_output):
        key = num_attn_0_1_output
        return 6

    num_mlp_0_0_outputs = [num_mlp_0_0(k0) for k0 in num_attn_0_1_outputs]
    num_mlp_0_0_output_scores = classifier_weights.loc[
        [("num_mlp_0_0_outputs", str(v)) for v in num_mlp_0_0_outputs]
    ]

    # num_mlp_0_1 #################################################
    def num_mlp_0_1(num_attn_0_1_output):
        key = num_attn_0_1_output
        return 4

    num_mlp_0_1_outputs = [num_mlp_0_1(k0) for k0 in num_attn_0_1_outputs]
    num_mlp_0_1_output_scores = classifier_weights.loc[
        [("num_mlp_0_1_outputs", str(v)) for v in num_mlp_0_1_outputs]
    ]

    # attn_1_0 ####################################################
    def predicate_1_0(q_position, k_position):
        if q_position in {0, 1}:
            return k_position == 1
        elif q_position in {2}:
            return k_position == 2
        elif q_position in {3, 4, 5}:
            return k_position == 3
        elif q_position in {6, 7, 9, 11, 13, 15}:
            return k_position == 5
        elif q_position in {8, 10, 12, 14}:
            return k_position == 7

    attn_1_0_pattern = select_closest(positions, positions, predicate_1_0)
    attn_1_0_outputs = aggregate(attn_1_0_pattern, mlp_0_1_outputs)
    attn_1_0_output_scores = classifier_weights.loc[
        [("attn_1_0_outputs", str(v)) for v in attn_1_0_outputs]
    ]

    # attn_1_1 ####################################################
    def predicate_1_1(q_position, k_position):
        if q_position in {0, 5}:
            return k_position == 4
        elif q_position in {1, 11, 9}:
            return k_position == 1
        elif q_position in {2, 3, 13, 15}:
            return k_position == 2
        elif q_position in {4}:
            return k_position == 3
        elif q_position in {6, 7, 8, 10, 12, 14}:
            return k_position == 5

    attn_1_1_pattern = select_closest(positions, positions, predicate_1_1)
    attn_1_1_outputs = aggregate(attn_1_1_pattern, mlp_0_0_outputs)
    attn_1_1_output_scores = classifier_weights.loc[
        [("attn_1_1_outputs", str(v)) for v in attn_1_1_outputs]
    ]

    # num_attn_1_0 ####################################################
    def num_predicate_1_0(q_mlp_0_0_output, k_mlp_0_0_output):
        if q_mlp_0_0_output in {0}:
            return k_mlp_0_0_output == 2
        elif q_mlp_0_0_output in {1, 2, 3, 4, 6, 7, 9, 11, 13, 15}:
            return k_mlp_0_0_output == 12
        elif q_mlp_0_0_output in {5}:
            return k_mlp_0_0_output == 13
        elif q_mlp_0_0_output in {8, 12, 14}:
            return k_mlp_0_0_output == 6
        elif q_mlp_0_0_output in {10}:
            return k_mlp_0_0_output == 4

    num_attn_1_0_pattern = select(mlp_0_0_outputs, mlp_0_0_outputs, num_predicate_1_0)
    num_attn_1_0_outputs = aggregate_sum(num_attn_1_0_pattern, num_attn_0_0_outputs)
    num_attn_1_0_output_scores = classifier_weights.loc[
        [("num_attn_1_0_outputs", "_") for v in num_attn_1_0_outputs]
    ].mul(num_attn_1_0_outputs, axis=0)

    # num_attn_1_1 ####################################################
    def num_predicate_1_1(num_mlp_0_0_output, mlp_0_0_output):
        if num_mlp_0_0_output in {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}:
            return mlp_0_0_output == 5

    num_attn_1_1_pattern = select(
        mlp_0_0_outputs, num_mlp_0_0_outputs, num_predicate_1_1
    )
    num_attn_1_1_outputs = aggregate_sum(num_attn_1_1_pattern, ones)
    num_attn_1_1_output_scores = classifier_weights.loc[
        [("num_attn_1_1_outputs", "_") for v in num_attn_1_1_outputs]
    ].mul(num_attn_1_1_outputs, axis=0)

    # mlp_1_0 #####################################################
    def mlp_1_0(attn_1_1_output, attn_1_0_output):
        key = (attn_1_1_output, attn_1_0_output)
        if key in {
            (0, 0),
            (0, 3),
            (0, 10),
            (0, 12),
            (1, 0),
            (1, 3),
            (1, 10),
            (1, 11),
            (1, 12),
            (1, 13),
            (3, 0),
            (4, 3),
            (7, 0),
            (7, 3),
            (8, 1),
            (9, 1),
            (10, 0),
            (10, 3),
            (11, 3),
            (12, 0),
            (13, 0),
            (13, 1),
            (13, 3),
            (14, 3),
            (15, 0),
            (15, 3),
            (15, 10),
            (15, 12),
        }:
            return 6
        elif key in {
            (0, 8),
            (1, 8),
            (2, 6),
            (2, 8),
            (2, 12),
            (3, 6),
            (3, 8),
            (3, 12),
            (4, 8),
            (5, 8),
            (6, 8),
            (7, 6),
            (7, 8),
            (8, 6),
            (8, 8),
            (10, 6),
            (10, 8),
            (11, 8),
            (12, 6),
            (12, 8),
            (13, 6),
            (13, 8),
            (14, 6),
            (14, 8),
            (15, 8),
        }:
            return 14
        elif key in {
            (3, 3),
            (3, 5),
            (5, 3),
            (5, 5),
            (5, 6),
            (5, 10),
            (5, 12),
            (5, 13),
            (6, 3),
            (6, 5),
            (6, 6),
            (6, 10),
            (6, 12),
            (6, 13),
            (8, 3),
            (8, 5),
            (12, 3),
            (12, 5),
            (12, 10),
            (12, 12),
        }:
            return 8
        elif key in {
            (0, 1),
            (1, 1),
            (2, 1),
            (3, 1),
            (4, 1),
            (5, 1),
            (10, 1),
            (11, 1),
            (14, 1),
            (15, 1),
        }:
            return 1
        return 10

    mlp_1_0_outputs = [
        mlp_1_0(k0, k1) for k0, k1 in zip(attn_1_1_outputs, attn_1_0_outputs)
    ]
    mlp_1_0_output_scores = classifier_weights.loc[
        [("mlp_1_0_outputs", str(v)) for v in mlp_1_0_outputs]
    ]

    # mlp_1_1 #####################################################
    def mlp_1_1(attn_1_1_output, attn_1_0_output):
        key = (attn_1_1_output, attn_1_0_output)
        if key in {
            (0, 0),
            (0, 1),
            (0, 3),
            (0, 4),
            (0, 9),
            (0, 11),
            (0, 13),
            (0, 14),
            (0, 15),
            (1, 0),
            (1, 1),
            (1, 3),
            (1, 4),
            (1, 9),
            (1, 11),
            (1, 13),
            (1, 14),
            (1, 15),
            (2, 0),
            (2, 1),
            (2, 13),
            (2, 14),
            (3, 0),
            (3, 1),
            (3, 4),
            (3, 11),
            (3, 13),
            (3, 14),
            (4, 0),
            (4, 1),
            (4, 4),
            (4, 11),
            (4, 13),
            (4, 14),
            (5, 0),
            (5, 1),
            (5, 4),
            (5, 11),
            (5, 13),
            (5, 14),
            (6, 1),
            (7, 0),
            (7, 1),
            (7, 4),
            (7, 9),
            (7, 11),
            (7, 13),
            (7, 14),
            (7, 15),
            (8, 0),
            (8, 1),
            (8, 3),
            (8, 4),
            (8, 9),
            (8, 11),
            (8, 13),
            (8, 14),
            (8, 15),
            (9, 0),
            (9, 1),
            (9, 13),
            (9, 14),
            (10, 0),
            (10, 1),
            (10, 13),
            (10, 14),
            (11, 0),
            (11, 1),
            (11, 4),
            (11, 11),
            (11, 13),
            (11, 14),
            (12, 0),
            (12, 1),
            (13, 0),
            (13, 1),
            (13, 4),
            (13, 9),
            (13, 11),
            (13, 13),
            (13, 14),
            (13, 15),
            (14, 0),
            (14, 1),
            (14, 4),
            (14, 9),
            (14, 11),
            (14, 13),
            (14, 14),
            (15, 0),
            (15, 1),
            (15, 4),
            (15, 9),
            (15, 11),
            (15, 13),
            (15, 14),
        }:
            return 7
        elif key in {(3, 3)}:
            return 0
        return 8

    mlp_1_1_outputs = [
        mlp_1_1(k0, k1) for k0, k1 in zip(attn_1_1_outputs, attn_1_0_outputs)
    ]
    mlp_1_1_output_scores = classifier_weights.loc[
        [("mlp_1_1_outputs", str(v)) for v in mlp_1_1_outputs]
    ]

    # num_mlp_1_0 #################################################
    def num_mlp_1_0(num_attn_0_1_output, num_attn_1_1_output):
        key = (num_attn_0_1_output, num_attn_1_1_output)
        return 4

    num_mlp_1_0_outputs = [
        num_mlp_1_0(k0, k1)
        for k0, k1 in zip(num_attn_0_1_outputs, num_attn_1_1_outputs)
    ]
    num_mlp_1_0_output_scores = classifier_weights.loc[
        [("num_mlp_1_0_outputs", str(v)) for v in num_mlp_1_0_outputs]
    ]

    # num_mlp_1_1 #################################################
    def num_mlp_1_1(num_attn_1_1_output):
        key = num_attn_1_1_output
        return 7

    num_mlp_1_1_outputs = [num_mlp_1_1(k0) for k0 in num_attn_1_1_outputs]
    num_mlp_1_1_output_scores = classifier_weights.loc[
        [("num_mlp_1_1_outputs", str(v)) for v in num_mlp_1_1_outputs]
    ]

    # attn_2_0 ####################################################
    def predicate_2_0(mlp_0_0_output, position):
        if mlp_0_0_output in {0, 4, 15}:
            return position == 1
        elif mlp_0_0_output in {1, 11, 7}:
            return position == 2
        elif mlp_0_0_output in {2, 5, 6, 8, 9, 10, 13, 14}:
            return position == 3
        elif mlp_0_0_output in {3, 12}:
            return position == 4

    attn_2_0_pattern = select_closest(positions, mlp_0_0_outputs, predicate_2_0)
    attn_2_0_outputs = aggregate(attn_2_0_pattern, attn_1_0_outputs)
    attn_2_0_output_scores = classifier_weights.loc[
        [("attn_2_0_outputs", str(v)) for v in attn_2_0_outputs]
    ]

    # attn_2_1 ####################################################
    def predicate_2_1(token, position):
        if token in {"{", "("}:
            return position == 3
        elif token in {"<s>", "}", ")"}:
            return position == 4

    attn_2_1_pattern = select_closest(positions, tokens, predicate_2_1)
    attn_2_1_outputs = aggregate(attn_2_1_pattern, attn_1_0_outputs)
    attn_2_1_output_scores = classifier_weights.loc[
        [("attn_2_1_outputs", str(v)) for v in attn_2_1_outputs]
    ]

    # num_attn_2_0 ####################################################
    def num_predicate_2_0(num_mlp_1_0_output, token):
        if num_mlp_1_0_output in {0, 2, 5, 7, 11, 13}:
            return token == "}"
        elif num_mlp_1_0_output in {1, 4, 12, 15}:
            return token == ""
        elif num_mlp_1_0_output in {3, 6, 8, 9, 10, 14}:
            return token == ")"

    num_attn_2_0_pattern = select(tokens, num_mlp_1_0_outputs, num_predicate_2_0)
    num_attn_2_0_outputs = aggregate_sum(num_attn_2_0_pattern, num_attn_1_0_outputs)
    num_attn_2_0_output_scores = classifier_weights.loc[
        [("num_attn_2_0_outputs", "_") for v in num_attn_2_0_outputs]
    ].mul(num_attn_2_0_outputs, axis=0)

    # num_attn_2_1 ####################################################
    def num_predicate_2_1(attn_1_0_output, mlp_0_1_output):
        if attn_1_0_output in {0, 4, 5, 6, 7, 8, 9, 10, 12, 14}:
            return mlp_0_1_output == 5
        elif attn_1_0_output in {1, 2, 3, 11, 13, 15}:
            return mlp_0_1_output == 13

    num_attn_2_1_pattern = select(mlp_0_1_outputs, attn_1_0_outputs, num_predicate_2_1)
    num_attn_2_1_outputs = aggregate_sum(num_attn_2_1_pattern, ones)
    num_attn_2_1_output_scores = classifier_weights.loc[
        [("num_attn_2_1_outputs", "_") for v in num_attn_2_1_outputs]
    ].mul(num_attn_2_1_outputs, axis=0)

    # mlp_2_0 #####################################################
    def mlp_2_0(attn_0_1_output, attn_2_0_output):
        key = (attn_0_1_output, attn_2_0_output)
        if key in {
            ("(", 1),
            ("(", 2),
            ("(", 3),
            ("(", 4),
            ("(", 6),
            ("(", 7),
            ("(", 9),
            ("(", 10),
            ("(", 11),
            ("(", 12),
            ("(", 13),
            ("(", 14),
            (")", 1),
            (")", 3),
            (")", 4),
            (")", 6),
            (")", 9),
            (")", 10),
            (")", 11),
            (")", 14),
            ("<s>", 1),
            ("<s>", 3),
            ("<s>", 4),
            ("<s>", 6),
            ("<s>", 7),
            ("<s>", 9),
            ("<s>", 10),
            ("<s>", 11),
            ("<s>", 12),
            ("<s>", 14),
            ("{", 0),
            ("{", 1),
            ("{", 2),
            ("{", 3),
            ("{", 4),
            ("{", 6),
            ("{", 7),
            ("{", 9),
            ("{", 10),
            ("{", 11),
            ("{", 12),
            ("{", 13),
            ("{", 14),
            ("{", 15),
        }:
            return 0
        elif key in {
            (")", 7),
            ("}", 0),
            ("}", 2),
            ("}", 3),
            ("}", 4),
            ("}", 7),
            ("}", 9),
            ("}", 10),
            ("}", 11),
            ("}", 12),
            ("}", 13),
            ("}", 14),
            ("}", 15),
        }:
            return 7
        elif key in {("}", 6)}:
            return 8
        return 11

    mlp_2_0_outputs = [
        mlp_2_0(k0, k1) for k0, k1 in zip(attn_0_1_outputs, attn_2_0_outputs)
    ]
    mlp_2_0_output_scores = classifier_weights.loc[
        [("mlp_2_0_outputs", str(v)) for v in mlp_2_0_outputs]
    ]

    # mlp_2_1 #####################################################
    def mlp_2_1(attn_2_0_output, attn_1_0_output):
        key = (attn_2_0_output, attn_1_0_output)
        if key in {
            (0, 12),
            (2, 7),
            (2, 10),
            (2, 12),
            (5, 5),
            (5, 7),
            (5, 10),
            (5, 12),
            (6, 0),
            (6, 2),
            (6, 5),
            (6, 6),
            (6, 7),
            (6, 9),
            (6, 10),
            (6, 11),
            (6, 12),
            (6, 13),
            (6, 15),
            (7, 7),
            (7, 10),
            (7, 12),
            (8, 0),
            (8, 2),
            (8, 5),
            (8, 6),
            (8, 7),
            (8, 10),
            (8, 12),
            (8, 13),
            (8, 15),
            (9, 7),
            (9, 12),
            (10, 5),
            (10, 7),
            (10, 10),
            (10, 12),
            (11, 7),
            (11, 12),
            (12, 0),
            (12, 2),
            (12, 5),
            (12, 6),
            (12, 7),
            (12, 8),
            (12, 9),
            (12, 10),
            (12, 11),
            (12, 12),
            (12, 13),
            (12, 15),
            (13, 5),
            (13, 10),
            (13, 12),
            (14, 0),
            (14, 2),
            (14, 5),
            (14, 6),
            (14, 7),
            (14, 10),
            (14, 12),
            (14, 13),
            (14, 15),
            (15, 5),
            (15, 7),
            (15, 12),
        }:
            return 0
        elif key in {
            (0, 1),
            (0, 3),
            (2, 1),
            (2, 3),
            (3, 1),
            (3, 3),
            (3, 9),
            (3, 10),
            (3, 12),
            (4, 1),
            (4, 3),
            (4, 12),
            (5, 1),
            (5, 3),
            (6, 1),
            (6, 3),
            (7, 1),
            (7, 3),
            (8, 1),
            (8, 3),
            (9, 1),
            (9, 3),
            (10, 1),
            (10, 3),
            (11, 1),
            (11, 3),
            (12, 1),
            (12, 3),
            (13, 1),
            (13, 3),
            (14, 1),
            (14, 3),
            (14, 9),
            (14, 11),
            (15, 1),
            (15, 3),
        }:
            return 2
        elif key in {
            (2, 14),
            (6, 8),
            (6, 14),
            (7, 8),
            (7, 14),
            (8, 8),
            (8, 9),
            (8, 11),
            (8, 14),
            (10, 14),
            (12, 14),
            (14, 8),
            (14, 14),
        }:
            return 7
        return 1

    mlp_2_1_outputs = [
        mlp_2_1(k0, k1) for k0, k1 in zip(attn_2_0_outputs, attn_1_0_outputs)
    ]
    mlp_2_1_output_scores = classifier_weights.loc[
        [("mlp_2_1_outputs", str(v)) for v in mlp_2_1_outputs]
    ]

    # num_mlp_2_0 #################################################
    def num_mlp_2_0(num_attn_0_1_output, num_attn_2_1_output):
        key = (num_attn_0_1_output, num_attn_2_1_output)
        return 13

    num_mlp_2_0_outputs = [
        num_mlp_2_0(k0, k1)
        for k0, k1 in zip(num_attn_0_1_outputs, num_attn_2_1_outputs)
    ]
    num_mlp_2_0_output_scores = classifier_weights.loc[
        [("num_mlp_2_0_outputs", str(v)) for v in num_mlp_2_0_outputs]
    ]

    # num_mlp_2_1 #################################################
    def num_mlp_2_1(num_attn_1_1_output):
        key = num_attn_1_1_output
        if key in {0}:
            return 3
        return 12

    num_mlp_2_1_outputs = [num_mlp_2_1(k0) for k0 in num_attn_1_1_outputs]
    num_mlp_2_1_output_scores = classifier_weights.loc[
        [("num_mlp_2_1_outputs", str(v)) for v in num_mlp_2_1_outputs]
    ]

    feature_logits = pd.concat(
        [
            df.reset_index()
            for df in [
                token_scores,
                position_scores,
                attn_0_0_output_scores,
                attn_0_1_output_scores,
                mlp_0_0_output_scores,
                mlp_0_1_output_scores,
                num_mlp_0_0_output_scores,
                num_mlp_0_1_output_scores,
                attn_1_0_output_scores,
                attn_1_1_output_scores,
                mlp_1_0_output_scores,
                mlp_1_1_output_scores,
                num_mlp_1_0_output_scores,
                num_mlp_1_1_output_scores,
                attn_2_0_output_scores,
                attn_2_1_output_scores,
                mlp_2_0_output_scores,
                mlp_2_1_output_scores,
                num_mlp_2_0_output_scores,
                num_mlp_2_1_output_scores,
                one_scores,
                num_attn_0_0_output_scores,
                num_attn_0_1_output_scores,
                num_attn_1_0_output_scores,
                num_attn_1_1_output_scores,
                num_attn_2_0_output_scores,
                num_attn_2_1_output_scores,
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
            "{",
            ")",
            ")",
            "(",
            "{",
            "}",
            "(",
            "}",
            "{",
            "(",
            "{",
            "{",
            "}",
            "}",
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
            "{",
            "(",
            "(",
            "{",
            "}",
            ")",
            ")",
            "}",
            "{",
            "}",
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
    (
        [
            "<s>",
            "{",
            "{",
            "}",
            "{",
            "}",
            "{",
            "}",
            "(",
            ")",
            "}",
            "{",
            "}",
            "{",
            "}",
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
            "T",
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
            "}",
            "{",
            ")",
            ")",
            "}",
            "{",
            "(",
            "}",
            "{",
            "}",
            "{",
            ")",
            "}",
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
            "{",
            "{",
            "{",
            "(",
            "(",
            "(",
            ")",
            ")",
            "(",
            ")",
            ")",
            "}",
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
            "{",
            "{",
            "{",
            "(",
            ")",
            "{",
            "}",
            "{",
            "}",
            "}",
            "(",
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
            "{",
            "{",
            "}",
            "(",
            ")",
            "{",
            "}",
            "}",
            "}",
            "(",
            "(",
            "}",
            ")",
            "{",
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
            "T",
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
            "{",
            "(",
            "(",
            ")",
            ")",
            "{",
            "}",
            "{",
            "}",
            "}",
            "}",
            "{",
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
            "{",
            "}",
            "(",
            ")",
            "(",
            ")",
            "(",
            ")",
            "(",
            ")",
            "(",
            ")",
            "{",
            "}",
            "(",
        ],
        [
            "<pad>",
            "P",
            "T",
            "P",
            "T",
            "P",
            "T",
            "P",
            "T",
            "P",
            "T",
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
            "{",
            "(",
            "(",
            "{",
            "}",
            ")",
            ")",
            "(",
            ")",
            "}",
            "(",
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
]
for x, y in examples:
    print(f"x: {x}")
    print(f"y: {y}")
    y_hat = run(x)
    print(f"y_hat: {y_hat}")
    print()
