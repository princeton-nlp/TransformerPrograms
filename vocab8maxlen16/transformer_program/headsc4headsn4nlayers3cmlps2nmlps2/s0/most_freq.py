import numpy as np
import pandas as pd


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


def select(keys, queries, predicate):
    return [[predicate(q, k) for k in keys] for q in queries]


def aggregate(attention, values):
    return [[v for a, v in zip(attn, values) if a][0] for attn in attention]


def aggregate_sum(attention, values):
    return [sum([v for a, v in zip(attn, values) if a]) for attn in attention]


def run(tokens):

    # classifier weights ##########################################
    classifier_weights = pd.read_csv(
        "output/rasp/sort/vocab8maxlen16/transformer_program/headsc4headsn4nlayers3cmlps2nmlps2/s0/most_freq_weights.csv",
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
    def predicate_0_0(position, token):
        if position in {0, 8}:
            return token == "0"
        elif position in {1, 2, 3, 7, 9, 10, 11, 12, 13, 14}:
            return token == "4"
        elif position in {4, 5, 6}:
            return token == "5"
        elif position in {15}:
            return token == "<s>"

    attn_0_0_pattern = select_closest(tokens, positions, predicate_0_0)
    attn_0_0_outputs = aggregate(attn_0_0_pattern, tokens)
    attn_0_0_output_scores = classifier_weights.loc[
        [("attn_0_0_outputs", str(v)) for v in attn_0_0_outputs]
    ]

    # attn_0_1 ####################################################
    def predicate_0_1(q_position, k_position):
        if q_position in {0, 14}:
            return k_position == 2
        elif q_position in {1, 2}:
            return k_position == 1
        elif q_position in {11, 10, 3, 7}:
            return k_position == 4
        elif q_position in {9, 4, 12, 15}:
            return k_position == 5
        elif q_position in {5}:
            return k_position == 6
        elif q_position in {13, 6}:
            return k_position == 7
        elif q_position in {8}:
            return k_position == 3

    attn_0_1_pattern = select_closest(positions, positions, predicate_0_1)
    attn_0_1_outputs = aggregate(attn_0_1_pattern, tokens)
    attn_0_1_output_scores = classifier_weights.loc[
        [("attn_0_1_outputs", str(v)) for v in attn_0_1_outputs]
    ]

    # attn_0_2 ####################################################
    def predicate_0_2(q_position, k_position):
        if q_position in {0, 15}:
            return k_position == 3
        elif q_position in {1}:
            return k_position == 7
        elif q_position in {2, 3, 4, 5}:
            return k_position == 1
        elif q_position in {11, 6, 14}:
            return k_position == 2
        elif q_position in {10, 7}:
            return k_position == 4
        elif q_position in {8, 12}:
            return k_position == 5
        elif q_position in {9}:
            return k_position == 6
        elif q_position in {13}:
            return k_position == 8

    attn_0_2_pattern = select_closest(positions, positions, predicate_0_2)
    attn_0_2_outputs = aggregate(attn_0_2_pattern, tokens)
    attn_0_2_output_scores = classifier_weights.loc[
        [("attn_0_2_outputs", str(v)) for v in attn_0_2_outputs]
    ]

    # attn_0_3 ####################################################
    def predicate_0_3(position, token):
        if position in {0, 4, 5, 6}:
            return token == "1"
        elif position in {1, 13, 7}:
            return token == "4"
        elif position in {11, 2, 3}:
            return token == "3"
        elif position in {8, 14}:
            return token == "0"
        elif position in {9}:
            return token == ""
        elif position in {10, 12, 15}:
            return token == "2"

    attn_0_3_pattern = select_closest(tokens, positions, predicate_0_3)
    attn_0_3_outputs = aggregate(attn_0_3_pattern, tokens)
    attn_0_3_output_scores = classifier_weights.loc[
        [("attn_0_3_outputs", str(v)) for v in attn_0_3_outputs]
    ]

    # num_attn_0_0 ####################################################
    def num_predicate_0_0(position, token):
        if position in {0, 3, 4, 5, 6}:
            return token == "<s>"
        elif position in {1, 2}:
            return token == "3"
        elif position in {7, 8, 9, 10, 11, 12, 14, 15}:
            return token == ""
        elif position in {13}:
            return token == "<pad>"

    num_attn_0_0_pattern = select(tokens, positions, num_predicate_0_0)
    num_attn_0_0_outputs = aggregate_sum(num_attn_0_0_pattern, ones)
    num_attn_0_0_output_scores = classifier_weights.loc[
        [("num_attn_0_0_outputs", "_") for v in num_attn_0_0_outputs]
    ].mul(num_attn_0_0_outputs, axis=0)

    # num_attn_0_1 ####################################################
    def num_predicate_0_1(position, token):
        if position in {0}:
            return token == "<s>"
        elif position in {8, 1, 7}:
            return token == "4"
        elif position in {2, 3, 9, 10, 11, 13, 14, 15}:
            return token == ""
        elif position in {12, 4, 5, 6}:
            return token == "2"

    num_attn_0_1_pattern = select(tokens, positions, num_predicate_0_1)
    num_attn_0_1_outputs = aggregate_sum(num_attn_0_1_pattern, ones)
    num_attn_0_1_output_scores = classifier_weights.loc[
        [("num_attn_0_1_outputs", "_") for v in num_attn_0_1_outputs]
    ].mul(num_attn_0_1_outputs, axis=0)

    # num_attn_0_2 ####################################################
    def num_predicate_0_2(position, token):
        if position in {0, 7, 8, 9, 12}:
            return token == "<s>"
        elif position in {1}:
            return token == "3"
        elif position in {2, 10, 11, 13, 15}:
            return token == ""
        elif position in {3, 4, 14}:
            return token == "<pad>"
        elif position in {5, 6}:
            return token == "1"

    num_attn_0_2_pattern = select(tokens, positions, num_predicate_0_2)
    num_attn_0_2_outputs = aggregate_sum(num_attn_0_2_pattern, ones)
    num_attn_0_2_output_scores = classifier_weights.loc[
        [("num_attn_0_2_outputs", "_") for v in num_attn_0_2_outputs]
    ].mul(num_attn_0_2_outputs, axis=0)

    # num_attn_0_3 ####################################################
    def num_predicate_0_3(position, token):
        if position in {0, 1, 7, 8, 9}:
            return token == "5"
        elif position in {2, 5, 6}:
            return token == "<s>"
        elif position in {3, 4, 10, 11, 12, 14, 15}:
            return token == ""
        elif position in {13}:
            return token == "<pad>"

    num_attn_0_3_pattern = select(tokens, positions, num_predicate_0_3)
    num_attn_0_3_outputs = aggregate_sum(num_attn_0_3_pattern, ones)
    num_attn_0_3_output_scores = classifier_weights.loc[
        [("num_attn_0_3_outputs", "_") for v in num_attn_0_3_outputs]
    ].mul(num_attn_0_3_outputs, axis=0)

    # mlp_0_0 #####################################################
    def mlp_0_0(position, attn_0_3_output):
        key = (position, attn_0_3_output)
        if key in {
            (0, "0"),
            (0, "2"),
            (0, "3"),
            (0, "4"),
            (0, "5"),
            (0, "<s>"),
            (1, "0"),
            (3, "0"),
            (3, "2"),
            (3, "3"),
            (3, "4"),
            (3, "5"),
            (3, "<s>"),
            (4, "0"),
            (4, "3"),
            (4, "4"),
            (4, "5"),
            (4, "<s>"),
            (5, "0"),
            (5, "2"),
            (5, "3"),
            (5, "4"),
            (5, "5"),
            (5, "<s>"),
            (6, "0"),
            (6, "3"),
            (6, "4"),
            (6, "5"),
            (6, "<s>"),
        }:
            return 3
        elif key in {
            (1, "2"),
            (1, "3"),
            (1, "4"),
            (1, "5"),
            (1, "<s>"),
            (4, "2"),
            (6, "2"),
        }:
            return 1
        elif key in {
            (2, "0"),
            (2, "1"),
            (2, "2"),
            (2, "3"),
            (2, "4"),
            (2, "5"),
            (2, "<s>"),
        }:
            return 4
        elif key in {(0, "1"), (1, "1"), (3, "1"), (4, "1"), (5, "1"), (6, "1")}:
            return 5
        return 12

    mlp_0_0_outputs = [mlp_0_0(k0, k1) for k0, k1 in zip(positions, attn_0_3_outputs)]
    mlp_0_0_output_scores = classifier_weights.loc[
        [("mlp_0_0_outputs", str(v)) for v in mlp_0_0_outputs]
    ]

    # mlp_0_1 #####################################################
    def mlp_0_1(position, attn_0_0_output):
        key = (position, attn_0_0_output)
        if key in {
            (0, "2"),
            (5, "0"),
            (5, "1"),
            (5, "2"),
            (5, "3"),
            (5, "5"),
            (5, "<s>"),
            (8, "0"),
            (8, "2"),
            (8, "3"),
            (8, "4"),
        }:
            return 12
        elif key in {(0, "1"), (0, "5"), (0, "<s>"), (8, "5"), (8, "<s>")}:
            return 2
        elif key in {(2, "4"), (3, "4"), (9, "4"), (10, "4"), (13, "4")}:
            return 14
        elif key in {(1, "0"), (1, "2"), (1, "3"), (1, "4"), (1, "5")}:
            return 4
        elif key in {
            (4, "2"),
            (4, "3"),
            (4, "<s>"),
            (7, "<s>"),
            (9, "<s>"),
            (10, "<s>"),
            (11, "<s>"),
            (12, "<s>"),
            (14, "<s>"),
            (15, "<s>"),
        }:
            return 15
        elif key in {(0, "0"), (0, "3"), (0, "4"), (5, "4"), (6, "4"), (12, "4")}:
            return 1
        elif key in {
            (2, "3"),
            (2, "<s>"),
            (3, "<s>"),
            (4, "5"),
            (11, "5"),
            (13, "<s>"),
        }:
            return 6
        elif key in {(1, "<s>")}:
            return 0
        return 3

    mlp_0_1_outputs = [mlp_0_1(k0, k1) for k0, k1 in zip(positions, attn_0_0_outputs)]
    mlp_0_1_output_scores = classifier_weights.loc[
        [("mlp_0_1_outputs", str(v)) for v in mlp_0_1_outputs]
    ]

    # num_mlp_0_0 #################################################
    def num_mlp_0_0(num_attn_0_1_output, num_attn_0_0_output):
        key = (num_attn_0_1_output, num_attn_0_0_output)
        return 11

    num_mlp_0_0_outputs = [
        num_mlp_0_0(k0, k1)
        for k0, k1 in zip(num_attn_0_1_outputs, num_attn_0_0_outputs)
    ]
    num_mlp_0_0_output_scores = classifier_weights.loc[
        [("num_mlp_0_0_outputs", str(v)) for v in num_mlp_0_0_outputs]
    ]

    # num_mlp_0_1 #################################################
    def num_mlp_0_1(num_attn_0_3_output, num_attn_0_2_output):
        key = (num_attn_0_3_output, num_attn_0_2_output)
        return 1

    num_mlp_0_1_outputs = [
        num_mlp_0_1(k0, k1)
        for k0, k1 in zip(num_attn_0_3_outputs, num_attn_0_2_outputs)
    ]
    num_mlp_0_1_output_scores = classifier_weights.loc[
        [("num_mlp_0_1_outputs", str(v)) for v in num_mlp_0_1_outputs]
    ]

    # attn_1_0 ####################################################
    def predicate_1_0(token, attn_0_1_output):
        if token in {"4", "<s>", "0", "3"}:
            return attn_0_1_output == ""
        elif token in {"1", "2"}:
            return attn_0_1_output == "3"
        elif token in {"5"}:
            return attn_0_1_output == "5"

    attn_1_0_pattern = select_closest(attn_0_1_outputs, tokens, predicate_1_0)
    attn_1_0_outputs = aggregate(attn_1_0_pattern, attn_0_1_outputs)
    attn_1_0_output_scores = classifier_weights.loc[
        [("attn_1_0_outputs", str(v)) for v in attn_1_0_outputs]
    ]

    # attn_1_1 ####################################################
    def predicate_1_1(token, mlp_0_0_output):
        if token in {"0"}:
            return mlp_0_0_output == 8
        elif token in {"1", "3", "5"}:
            return mlp_0_0_output == 12
        elif token in {"4", "<s>", "2"}:
            return mlp_0_0_output == 5

    attn_1_1_pattern = select_closest(mlp_0_0_outputs, tokens, predicate_1_1)
    attn_1_1_outputs = aggregate(attn_1_1_pattern, mlp_0_0_outputs)
    attn_1_1_output_scores = classifier_weights.loc[
        [("attn_1_1_outputs", str(v)) for v in attn_1_1_outputs]
    ]

    # attn_1_2 ####################################################
    def predicate_1_2(q_token, k_token):
        if q_token in {"1", "0", "3", "5"}:
            return k_token == "2"
        elif q_token in {"2"}:
            return k_token == "3"
        elif q_token in {"4"}:
            return k_token == "<s>"
        elif q_token in {"<s>"}:
            return k_token == "1"

    attn_1_2_pattern = select_closest(tokens, tokens, predicate_1_2)
    attn_1_2_outputs = aggregate(attn_1_2_pattern, mlp_0_1_outputs)
    attn_1_2_output_scores = classifier_weights.loc[
        [("attn_1_2_outputs", str(v)) for v in attn_1_2_outputs]
    ]

    # attn_1_3 ####################################################
    def predicate_1_3(attn_0_0_output, position):
        if attn_0_0_output in {"0"}:
            return position == 3
        elif attn_0_0_output in {"2", "1", "5", "<s>", "3"}:
            return position == 0
        elif attn_0_0_output in {"4"}:
            return position == 8

    attn_1_3_pattern = select_closest(positions, attn_0_0_outputs, predicate_1_3)
    attn_1_3_outputs = aggregate(attn_1_3_pattern, mlp_0_1_outputs)
    attn_1_3_output_scores = classifier_weights.loc[
        [("attn_1_3_outputs", str(v)) for v in attn_1_3_outputs]
    ]

    # num_attn_1_0 ####################################################
    def num_predicate_1_0(attn_0_0_output, attn_0_3_output):
        if attn_0_0_output in {"0", "2", "1", "<s>", "3"}:
            return attn_0_3_output == "5"
        elif attn_0_0_output in {"4", "5"}:
            return attn_0_3_output == "3"

    num_attn_1_0_pattern = select(attn_0_3_outputs, attn_0_0_outputs, num_predicate_1_0)
    num_attn_1_0_outputs = aggregate_sum(num_attn_1_0_pattern, ones)
    num_attn_1_0_output_scores = classifier_weights.loc[
        [("num_attn_1_0_outputs", "_") for v in num_attn_1_0_outputs]
    ].mul(num_attn_1_0_outputs, axis=0)

    # num_attn_1_1 ####################################################
    def num_predicate_1_1(position, token):
        if position in {0}:
            return token == "0"
        elif position in {1}:
            return token == "2"
        elif position in {2, 3, 4}:
            return token == "<s>"
        elif position in {5, 6}:
            return token == "1"
        elif position in {7, 8, 9, 10, 11, 13, 15}:
            return token == ""
        elif position in {12, 14}:
            return token == "<pad>"

    num_attn_1_1_pattern = select(tokens, positions, num_predicate_1_1)
    num_attn_1_1_outputs = aggregate_sum(num_attn_1_1_pattern, ones)
    num_attn_1_1_output_scores = classifier_weights.loc[
        [("num_attn_1_1_outputs", "_") for v in num_attn_1_1_outputs]
    ].mul(num_attn_1_1_outputs, axis=0)

    # num_attn_1_2 ####################################################
    def num_predicate_1_2(attn_0_0_output, attn_0_2_output):
        if attn_0_0_output in {"<s>", "0", "3"}:
            return attn_0_2_output == ""
        elif attn_0_0_output in {"1", "2", "4"}:
            return attn_0_2_output == "<pad>"
        elif attn_0_0_output in {"5"}:
            return attn_0_2_output == "5"

    num_attn_1_2_pattern = select(attn_0_2_outputs, attn_0_0_outputs, num_predicate_1_2)
    num_attn_1_2_outputs = aggregate_sum(num_attn_1_2_pattern, num_attn_0_3_outputs)
    num_attn_1_2_output_scores = classifier_weights.loc[
        [("num_attn_1_2_outputs", "_") for v in num_attn_1_2_outputs]
    ].mul(num_attn_1_2_outputs, axis=0)

    # num_attn_1_3 ####################################################
    def num_predicate_1_3(position, token):
        if position in {0}:
            return token == "0"
        elif position in {1}:
            return token == "1"
        elif position in {2, 3, 4, 5, 6}:
            return token == "<s>"
        elif position in {7, 8, 11, 12, 14}:
            return token == ""
        elif position in {9}:
            return token == "4"
        elif position in {10, 13, 15}:
            return token == "<pad>"

    num_attn_1_3_pattern = select(tokens, positions, num_predicate_1_3)
    num_attn_1_3_outputs = aggregate_sum(num_attn_1_3_pattern, ones)
    num_attn_1_3_output_scores = classifier_weights.loc[
        [("num_attn_1_3_outputs", "_") for v in num_attn_1_3_outputs]
    ].mul(num_attn_1_3_outputs, axis=0)

    # mlp_1_0 #####################################################
    def mlp_1_0(attn_0_0_output, token):
        key = (attn_0_0_output, token)
        if key in {("3", "3"), ("5", "3")}:
            return 13
        return 2

    mlp_1_0_outputs = [mlp_1_0(k0, k1) for k0, k1 in zip(attn_0_0_outputs, tokens)]
    mlp_1_0_output_scores = classifier_weights.loc[
        [("mlp_1_0_outputs", str(v)) for v in mlp_1_0_outputs]
    ]

    # mlp_1_1 #####################################################
    def mlp_1_1(attn_1_0_output, position):
        key = (attn_1_0_output, position)
        if key in {
            ("0", 6),
            ("1", 0),
            ("1", 6),
            ("2", 6),
            ("3", 0),
            ("3", 6),
            ("3", 7),
            ("3", 8),
            ("3", 9),
            ("3", 10),
            ("3", 11),
            ("3", 12),
            ("3", 13),
            ("3", 14),
            ("3", 15),
            ("4", 0),
            ("4", 6),
            ("5", 0),
            ("5", 6),
            ("<s>", 0),
            ("<s>", 6),
        }:
            return 11
        elif key in {
            ("0", 0),
            ("0", 1),
            ("0", 2),
            ("1", 1),
            ("1", 2),
            ("2", 0),
            ("2", 1),
            ("2", 2),
            ("3", 1),
            ("3", 2),
            ("4", 2),
            ("<s>", 1),
            ("<s>", 2),
        }:
            return 4
        elif key in {
            ("0", 4),
            ("0", 5),
            ("1", 4),
            ("1", 5),
            ("2", 4),
            ("2", 5),
            ("4", 4),
            ("4", 5),
            ("5", 4),
            ("5", 5),
            ("5", 14),
            ("<s>", 4),
            ("<s>", 5),
        }:
            return 2
        elif key in {("1", 3), ("3", 3), ("3", 5), ("4", 3), ("5", 3), ("<s>", 3)}:
            return 3
        elif key in {("0", 3), ("2", 3), ("3", 4), ("5", 2)}:
            return 6
        elif key in {("4", 1), ("5", 1)}:
            return 10
        return 1

    mlp_1_1_outputs = [mlp_1_1(k0, k1) for k0, k1 in zip(attn_1_0_outputs, positions)]
    mlp_1_1_output_scores = classifier_weights.loc[
        [("mlp_1_1_outputs", str(v)) for v in mlp_1_1_outputs]
    ]

    # num_mlp_1_0 #################################################
    def num_mlp_1_0(num_attn_0_1_output, num_attn_0_0_output):
        key = (num_attn_0_1_output, num_attn_0_0_output)
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
            (1, 1),
            (1, 2),
            (1, 3),
            (1, 4),
            (1, 5),
            (1, 6),
            (1, 7),
            (1, 8),
            (1, 9),
            (1, 10),
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
            (2, 2),
            (2, 3),
            (2, 4),
            (2, 5),
            (2, 6),
            (2, 7),
            (2, 8),
            (2, 9),
            (2, 10),
            (2, 11),
            (2, 12),
            (2, 13),
            (2, 14),
            (2, 15),
            (2, 16),
            (2, 17),
            (2, 18),
            (2, 19),
            (2, 20),
            (2, 21),
            (2, 22),
            (2, 23),
            (2, 24),
            (2, 25),
            (2, 26),
            (2, 27),
            (2, 28),
            (2, 29),
            (2, 30),
            (2, 31),
            (3, 3),
            (3, 4),
            (3, 5),
            (3, 6),
            (3, 7),
            (3, 8),
            (3, 9),
            (3, 10),
            (3, 11),
            (3, 12),
            (3, 13),
            (3, 14),
            (3, 15),
            (3, 16),
            (3, 17),
            (3, 18),
            (3, 19),
            (3, 20),
            (3, 21),
            (3, 22),
            (3, 23),
            (3, 24),
            (3, 25),
            (3, 26),
            (3, 27),
            (3, 28),
            (3, 29),
            (3, 30),
            (3, 31),
            (4, 4),
            (4, 5),
            (4, 6),
            (4, 7),
            (4, 8),
            (4, 9),
            (4, 10),
            (4, 11),
            (4, 12),
            (4, 13),
            (4, 14),
            (4, 15),
            (4, 16),
            (4, 17),
            (4, 18),
            (4, 19),
            (4, 20),
            (4, 21),
            (4, 22),
            (4, 23),
            (4, 24),
            (4, 25),
            (4, 26),
            (4, 27),
            (4, 28),
            (4, 29),
            (4, 30),
            (4, 31),
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
            (5, 16),
            (5, 17),
            (5, 18),
            (5, 19),
            (5, 20),
            (5, 21),
            (5, 22),
            (5, 23),
            (5, 24),
            (5, 25),
            (5, 26),
            (5, 27),
            (5, 28),
            (5, 29),
            (5, 30),
            (5, 31),
            (6, 6),
            (6, 7),
            (6, 8),
            (6, 9),
            (6, 10),
            (6, 11),
            (6, 12),
            (6, 13),
            (6, 14),
            (6, 15),
            (6, 16),
            (6, 17),
            (6, 18),
            (6, 19),
            (6, 20),
            (6, 21),
            (6, 22),
            (6, 23),
            (6, 24),
            (6, 25),
            (6, 26),
            (6, 27),
            (6, 28),
            (6, 29),
            (6, 30),
            (6, 31),
            (7, 7),
            (7, 8),
            (7, 9),
            (7, 10),
            (7, 11),
            (7, 12),
            (7, 13),
            (7, 14),
            (7, 15),
            (7, 16),
            (7, 17),
            (7, 18),
            (7, 19),
            (7, 20),
            (7, 21),
            (7, 22),
            (7, 23),
            (7, 24),
            (7, 25),
            (7, 26),
            (7, 27),
            (7, 28),
            (7, 29),
            (7, 30),
            (7, 31),
            (8, 8),
            (8, 9),
            (8, 10),
            (8, 11),
            (8, 12),
            (8, 13),
            (8, 14),
            (8, 15),
            (8, 16),
            (8, 17),
            (8, 18),
            (8, 19),
            (8, 20),
            (8, 21),
            (8, 22),
            (8, 23),
            (8, 24),
            (8, 25),
            (8, 26),
            (8, 27),
            (8, 28),
            (8, 29),
            (8, 30),
            (8, 31),
            (9, 10),
            (9, 11),
            (9, 12),
            (9, 13),
            (9, 14),
            (9, 15),
            (9, 16),
            (9, 17),
            (9, 18),
            (9, 19),
            (9, 20),
            (9, 21),
            (9, 22),
            (9, 23),
            (9, 24),
            (9, 25),
            (9, 26),
            (9, 27),
            (9, 28),
            (9, 29),
            (9, 30),
            (9, 31),
            (10, 11),
            (10, 12),
            (10, 13),
            (10, 14),
            (10, 15),
            (10, 16),
            (10, 17),
            (10, 18),
            (10, 19),
            (10, 20),
            (10, 21),
            (10, 22),
            (10, 23),
            (10, 24),
            (10, 25),
            (10, 26),
            (10, 27),
            (10, 28),
            (10, 29),
            (10, 30),
            (10, 31),
            (11, 12),
            (11, 13),
            (11, 14),
            (11, 15),
            (11, 16),
            (11, 17),
            (11, 18),
            (11, 19),
            (11, 20),
            (11, 21),
            (11, 22),
            (11, 23),
            (11, 24),
            (11, 25),
            (11, 26),
            (11, 27),
            (11, 28),
            (11, 29),
            (11, 30),
            (11, 31),
            (12, 13),
            (12, 14),
            (12, 15),
            (12, 16),
            (12, 17),
            (12, 18),
            (12, 19),
            (12, 20),
            (12, 21),
            (12, 22),
            (12, 23),
            (12, 24),
            (12, 25),
            (12, 26),
            (12, 27),
            (12, 28),
            (12, 29),
            (12, 30),
            (12, 31),
            (13, 14),
            (13, 15),
            (13, 16),
            (13, 17),
            (13, 18),
            (13, 19),
            (13, 20),
            (13, 21),
            (13, 22),
            (13, 23),
            (13, 24),
            (13, 25),
            (13, 26),
            (13, 27),
            (13, 28),
            (13, 29),
            (13, 30),
            (13, 31),
            (14, 15),
            (14, 16),
            (14, 17),
            (14, 18),
            (14, 19),
            (14, 20),
            (14, 21),
            (14, 22),
            (14, 23),
            (14, 24),
            (14, 25),
            (14, 26),
            (14, 27),
            (14, 28),
            (14, 29),
            (14, 30),
            (14, 31),
            (15, 16),
            (15, 17),
            (15, 18),
            (15, 19),
            (15, 20),
            (15, 21),
            (15, 22),
            (15, 23),
            (15, 24),
            (15, 25),
            (15, 26),
            (15, 27),
            (15, 28),
            (15, 29),
            (15, 30),
            (15, 31),
            (16, 17),
            (16, 18),
            (16, 19),
            (16, 20),
            (16, 21),
            (16, 22),
            (16, 23),
            (16, 24),
            (16, 25),
            (16, 26),
            (16, 27),
            (16, 28),
            (16, 29),
            (16, 30),
            (16, 31),
            (17, 19),
            (17, 20),
            (17, 21),
            (17, 22),
            (17, 23),
            (17, 24),
            (17, 25),
            (17, 26),
            (17, 27),
            (17, 28),
            (17, 29),
            (17, 30),
            (17, 31),
            (18, 20),
            (18, 21),
            (18, 22),
            (18, 23),
            (18, 24),
            (18, 25),
            (18, 26),
            (18, 27),
            (18, 28),
            (18, 29),
            (18, 30),
            (18, 31),
            (19, 21),
            (19, 22),
            (19, 23),
            (19, 24),
            (19, 25),
            (19, 26),
            (19, 27),
            (19, 28),
            (19, 29),
            (19, 30),
            (19, 31),
            (20, 22),
            (20, 23),
            (20, 24),
            (20, 25),
            (20, 26),
            (20, 27),
            (20, 28),
            (20, 29),
            (20, 30),
            (20, 31),
            (21, 23),
            (21, 24),
            (21, 25),
            (21, 26),
            (21, 27),
            (21, 28),
            (21, 29),
            (21, 30),
            (21, 31),
            (22, 24),
            (22, 25),
            (22, 26),
            (22, 27),
            (22, 28),
            (22, 29),
            (22, 30),
            (22, 31),
            (23, 25),
            (23, 26),
            (23, 27),
            (23, 28),
            (23, 29),
            (23, 30),
            (23, 31),
            (24, 26),
            (24, 27),
            (24, 28),
            (24, 29),
            (24, 30),
            (24, 31),
            (25, 27),
            (25, 28),
            (25, 29),
            (25, 30),
            (25, 31),
            (26, 29),
            (26, 30),
            (26, 31),
            (27, 30),
            (27, 31),
            (28, 31),
        }:
            return 8
        elif key in {
            (1, 0),
            (2, 1),
            (3, 2),
            (4, 3),
            (5, 4),
            (6, 5),
            (9, 9),
            (10, 10),
            (11, 11),
            (12, 12),
            (13, 13),
            (14, 14),
            (15, 15),
            (17, 18),
            (18, 19),
            (19, 20),
            (20, 21),
            (21, 22),
            (22, 23),
            (23, 24),
            (26, 28),
            (27, 29),
            (28, 30),
            (29, 31),
        }:
            return 10
        return 14

    num_mlp_1_0_outputs = [
        num_mlp_1_0(k0, k1)
        for k0, k1 in zip(num_attn_0_1_outputs, num_attn_0_0_outputs)
    ]
    num_mlp_1_0_output_scores = classifier_weights.loc[
        [("num_mlp_1_0_outputs", str(v)) for v in num_mlp_1_0_outputs]
    ]

    # num_mlp_1_1 #################################################
    def num_mlp_1_1(num_attn_1_2_output, num_attn_0_1_output):
        key = (num_attn_1_2_output, num_attn_0_1_output)
        return 9

    num_mlp_1_1_outputs = [
        num_mlp_1_1(k0, k1)
        for k0, k1 in zip(num_attn_1_2_outputs, num_attn_0_1_outputs)
    ]
    num_mlp_1_1_output_scores = classifier_weights.loc[
        [("num_mlp_1_1_outputs", str(v)) for v in num_mlp_1_1_outputs]
    ]

    # attn_2_0 ####################################################
    def predicate_2_0(token, mlp_0_0_output):
        if token in {"4", "0", "3", "5"}:
            return mlp_0_0_output == 12
        elif token in {"1", "2"}:
            return mlp_0_0_output == 3
        elif token in {"<s>"}:
            return mlp_0_0_output == 10

    attn_2_0_pattern = select_closest(mlp_0_0_outputs, tokens, predicate_2_0)
    attn_2_0_outputs = aggregate(attn_2_0_pattern, attn_0_1_outputs)
    attn_2_0_output_scores = classifier_weights.loc[
        [("attn_2_0_outputs", str(v)) for v in attn_2_0_outputs]
    ]

    # attn_2_1 ####################################################
    def predicate_2_1(num_mlp_1_0_output, attn_1_3_output):
        if num_mlp_1_0_output in {0, 3, 5, 6, 7, 11}:
            return attn_1_3_output == 11
        elif num_mlp_1_0_output in {1, 2, 4, 9, 10, 12, 14, 15}:
            return attn_1_3_output == 15
        elif num_mlp_1_0_output in {8}:
            return attn_1_3_output == 1
        elif num_mlp_1_0_output in {13}:
            return attn_1_3_output == 7

    attn_2_1_pattern = select_closest(
        attn_1_3_outputs, num_mlp_1_0_outputs, predicate_2_1
    )
    attn_2_1_outputs = aggregate(attn_2_1_pattern, attn_1_3_outputs)
    attn_2_1_output_scores = classifier_weights.loc[
        [("attn_2_1_outputs", str(v)) for v in attn_2_1_outputs]
    ]

    # attn_2_2 ####################################################
    def predicate_2_2(token, mlp_0_0_output):
        if token in {"0", "2", "4", "1", "5", "3"}:
            return mlp_0_0_output == 5
        elif token in {"<s>"}:
            return mlp_0_0_output == 8

    attn_2_2_pattern = select_closest(mlp_0_0_outputs, tokens, predicate_2_2)
    attn_2_2_outputs = aggregate(attn_2_2_pattern, mlp_0_0_outputs)
    attn_2_2_output_scores = classifier_weights.loc[
        [("attn_2_2_outputs", str(v)) for v in attn_2_2_outputs]
    ]

    # attn_2_3 ####################################################
    def predicate_2_3(mlp_0_1_output, attn_1_3_output):
        if mlp_0_1_output in {0, 2, 5, 10, 12}:
            return attn_1_3_output == 11
        elif mlp_0_1_output in {1, 3, 6}:
            return attn_1_3_output == 15
        elif mlp_0_1_output in {9, 11, 4, 13}:
            return attn_1_3_output == 12
        elif mlp_0_1_output in {7}:
            return attn_1_3_output == 6
        elif mlp_0_1_output in {8}:
            return attn_1_3_output == 5
        elif mlp_0_1_output in {14}:
            return attn_1_3_output == 14
        elif mlp_0_1_output in {15}:
            return attn_1_3_output == 0

    attn_2_3_pattern = select_closest(attn_1_3_outputs, mlp_0_1_outputs, predicate_2_3)
    attn_2_3_outputs = aggregate(attn_2_3_pattern, attn_1_3_outputs)
    attn_2_3_output_scores = classifier_weights.loc[
        [("attn_2_3_outputs", str(v)) for v in attn_2_3_outputs]
    ]

    # num_attn_2_0 ####################################################
    def num_predicate_2_0(position, token):
        if position in {0, 5, 6}:
            return token == "1"
        elif position in {1}:
            return token == "0"
        elif position in {2, 3, 4}:
            return token == "<s>"
        elif position in {7, 8, 9, 12, 13, 14, 15}:
            return token == ""
        elif position in {10, 11}:
            return token == "<pad>"

    num_attn_2_0_pattern = select(tokens, positions, num_predicate_2_0)
    num_attn_2_0_outputs = aggregate_sum(num_attn_2_0_pattern, ones)
    num_attn_2_0_output_scores = classifier_weights.loc[
        [("num_attn_2_0_outputs", "_") for v in num_attn_2_0_outputs]
    ].mul(num_attn_2_0_outputs, axis=0)

    # num_attn_2_1 ####################################################
    def num_predicate_2_1(position, attn_0_3_output):
        if position in {0, 1, 2, 3}:
            return attn_0_3_output == ""
        elif position in {4, 5, 6, 7, 8, 11, 12, 14}:
            return attn_0_3_output == "4"
        elif position in {9, 10, 13}:
            return attn_0_3_output == "5"
        elif position in {15}:
            return attn_0_3_output == "3"

    num_attn_2_1_pattern = select(attn_0_3_outputs, positions, num_predicate_2_1)
    num_attn_2_1_outputs = aggregate_sum(num_attn_2_1_pattern, num_attn_0_1_outputs)
    num_attn_2_1_output_scores = classifier_weights.loc[
        [("num_attn_2_1_outputs", "_") for v in num_attn_2_1_outputs]
    ].mul(num_attn_2_1_outputs, axis=0)

    # num_attn_2_2 ####################################################
    def num_predicate_2_2(position, attn_0_0_output):
        if position in {0, 2, 3, 10, 12, 13, 15}:
            return attn_0_0_output == ""
        elif position in {1, 9, 7}:
            return attn_0_0_output == "4"
        elif position in {11, 4, 5, 6}:
            return attn_0_0_output == "0"
        elif position in {8, 14}:
            return attn_0_0_output == "<pad>"

    num_attn_2_2_pattern = select(attn_0_0_outputs, positions, num_predicate_2_2)
    num_attn_2_2_outputs = aggregate_sum(num_attn_2_2_pattern, num_attn_1_3_outputs)
    num_attn_2_2_output_scores = classifier_weights.loc[
        [("num_attn_2_2_outputs", "_") for v in num_attn_2_2_outputs]
    ].mul(num_attn_2_2_outputs, axis=0)

    # num_attn_2_3 ####################################################
    def num_predicate_2_3(attn_1_0_output, attn_0_3_output):
        if attn_1_0_output in {"4", "0", "5"}:
            return attn_0_3_output == ""
        elif attn_1_0_output in {"1", "2"}:
            return attn_0_3_output == "<pad>"
        elif attn_1_0_output in {"<s>", "3"}:
            return attn_0_3_output == "3"

    num_attn_2_3_pattern = select(attn_0_3_outputs, attn_1_0_outputs, num_predicate_2_3)
    num_attn_2_3_outputs = aggregate_sum(num_attn_2_3_pattern, num_attn_0_0_outputs)
    num_attn_2_3_output_scores = classifier_weights.loc[
        [("num_attn_2_3_outputs", "_") for v in num_attn_2_3_outputs]
    ].mul(num_attn_2_3_outputs, axis=0)

    # mlp_2_0 #####################################################
    def mlp_2_0(attn_2_2_output, position):
        key = (attn_2_2_output, position)
        return 14

    mlp_2_0_outputs = [mlp_2_0(k0, k1) for k0, k1 in zip(attn_2_2_outputs, positions)]
    mlp_2_0_output_scores = classifier_weights.loc[
        [("mlp_2_0_outputs", str(v)) for v in mlp_2_0_outputs]
    ]

    # mlp_2_1 #####################################################
    def mlp_2_1(position, mlp_0_0_output):
        key = (position, mlp_0_0_output)
        if key in {
            (0, 3),
            (0, 11),
            (0, 12),
            (6, 0),
            (6, 1),
            (6, 2),
            (6, 3),
            (6, 4),
            (6, 5),
            (6, 6),
            (6, 7),
            (6, 8),
            (6, 9),
            (6, 10),
            (6, 11),
            (6, 12),
            (6, 13),
            (6, 14),
            (6, 15),
            (7, 1),
            (7, 3),
            (7, 6),
            (7, 11),
            (7, 12),
            (8, 1),
            (8, 3),
            (8, 11),
            (8, 12),
            (9, 1),
            (9, 3),
            (9, 5),
            (9, 6),
            (9, 11),
            (9, 12),
            (9, 14),
            (9, 15),
            (10, 0),
            (10, 1),
            (10, 2),
            (10, 3),
            (10, 5),
            (10, 6),
            (10, 7),
            (10, 8),
            (10, 9),
            (10, 11),
            (10, 12),
            (10, 13),
            (10, 14),
            (10, 15),
            (11, 1),
            (11, 3),
            (11, 5),
            (11, 6),
            (11, 11),
            (11, 12),
            (11, 14),
            (11, 15),
            (12, 0),
            (12, 1),
            (12, 2),
            (12, 3),
            (12, 5),
            (12, 6),
            (12, 11),
            (12, 12),
            (12, 14),
            (12, 15),
            (13, 1),
            (13, 3),
            (13, 5),
            (13, 6),
            (13, 11),
            (13, 12),
            (13, 15),
            (14, 3),
            (14, 11),
            (14, 12),
            (15, 1),
            (15, 2),
            (15, 3),
            (15, 5),
            (15, 6),
            (15, 11),
            (15, 12),
            (15, 14),
            (15, 15),
        }:
            return 0
        elif key in {
            (1, 0),
            (1, 1),
            (1, 2),
            (1, 3),
            (1, 5),
            (1, 6),
            (1, 7),
            (1, 8),
            (1, 9),
            (1, 10),
            (1, 13),
            (1, 14),
            (1, 15),
            (2, 1),
        }:
            return 11
        return 5

    mlp_2_1_outputs = [mlp_2_1(k0, k1) for k0, k1 in zip(positions, mlp_0_0_outputs)]
    mlp_2_1_output_scores = classifier_weights.loc[
        [("mlp_2_1_outputs", str(v)) for v in mlp_2_1_outputs]
    ]

    # num_mlp_2_0 #################################################
    def num_mlp_2_0(num_attn_1_3_output, num_attn_1_2_output):
        key = (num_attn_1_3_output, num_attn_1_2_output)
        return 3

    num_mlp_2_0_outputs = [
        num_mlp_2_0(k0, k1)
        for k0, k1 in zip(num_attn_1_3_outputs, num_attn_1_2_outputs)
    ]
    num_mlp_2_0_output_scores = classifier_weights.loc[
        [("num_mlp_2_0_outputs", str(v)) for v in num_mlp_2_0_outputs]
    ]

    # num_mlp_2_1 #################################################
    def num_mlp_2_1(num_attn_2_1_output, num_attn_1_2_output):
        key = (num_attn_2_1_output, num_attn_1_2_output)
        return 7

    num_mlp_2_1_outputs = [
        num_mlp_2_1(k0, k1)
        for k0, k1 in zip(num_attn_2_1_outputs, num_attn_1_2_outputs)
    ]
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
                attn_0_2_output_scores,
                attn_0_3_output_scores,
                mlp_0_0_output_scores,
                mlp_0_1_output_scores,
                num_mlp_0_0_output_scores,
                num_mlp_0_1_output_scores,
                attn_1_0_output_scores,
                attn_1_1_output_scores,
                attn_1_2_output_scores,
                attn_1_3_output_scores,
                mlp_1_0_output_scores,
                mlp_1_1_output_scores,
                num_mlp_1_0_output_scores,
                num_mlp_1_1_output_scores,
                attn_2_0_output_scores,
                attn_2_1_output_scores,
                attn_2_2_output_scores,
                attn_2_3_output_scores,
                mlp_2_0_output_scores,
                mlp_2_1_output_scores,
                num_mlp_2_0_output_scores,
                num_mlp_2_1_output_scores,
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


print(run(["<s>", "0", "2", "5", "2", "4", "3", "5", "4", "5", "4", "5", "0", "5"]))
