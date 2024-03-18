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
        "programs/rasp/sort/sort_weights.csv", index_col=[0, 1], dtype={"feature": str}
    )
    # inputs #####################################################
    token_scores = classifier_weights.loc[[("tokens", str(v)) for v in tokens]]

    positions = list(range(len(tokens)))
    position_scores = classifier_weights.loc[[("positions", str(v)) for v in positions]]

    ones = [1 for _ in range(len(tokens))]
    one_scores = classifier_weights.loc[[("ones", "_") for v in ones]].mul(ones, axis=0)

    # attn_0_0 ####################################################
    def predicate_0_0(q_position, k_position):
        if q_position in {0, 6}:
            return k_position == 2
        elif q_position in {1}:
            return k_position == 4
        elif q_position in {2}:
            return k_position == 5
        elif q_position in {3, 5}:
            return k_position == 1
        elif q_position in {4}:
            return k_position == 3
        elif q_position in {7}:
            return k_position == 6

    attn_0_0_pattern = select_closest(positions, positions, predicate_0_0)
    attn_0_0_outputs = aggregate(attn_0_0_pattern, tokens)
    attn_0_0_output_scores = classifier_weights.loc[
        [("attn_0_0_outputs", str(v)) for v in attn_0_0_outputs]
    ]

    # attn_0_1 ####################################################
    def predicate_0_1(q_position, k_position):
        if q_position in {0, 1}:
            return k_position == 2
        elif q_position in {2}:
            return k_position == 3
        elif q_position in {3}:
            return k_position == 4
        elif q_position in {4}:
            return k_position == 5
        elif q_position in {5, 7}:
            return k_position == 6
        elif q_position in {6}:
            return k_position == 1

    attn_0_1_pattern = select_closest(positions, positions, predicate_0_1)
    attn_0_1_outputs = aggregate(attn_0_1_pattern, tokens)
    attn_0_1_output_scores = classifier_weights.loc[
        [("attn_0_1_outputs", str(v)) for v in attn_0_1_outputs]
    ]

    # attn_0_2 ####################################################
    def predicate_0_2(position, token):
        if position in {0, 7}:
            return token == "1"
        elif position in {1}:
            return token == "0"
        elif position in {2, 3}:
            return token == "3"
        elif position in {4, 5, 6}:
            return token == "4"

    attn_0_2_pattern = select_closest(tokens, positions, predicate_0_2)
    attn_0_2_outputs = aggregate(attn_0_2_pattern, tokens)
    attn_0_2_output_scores = classifier_weights.loc[
        [("attn_0_2_outputs", str(v)) for v in attn_0_2_outputs]
    ]

    # attn_0_3 ####################################################
    def predicate_0_3(q_position, k_position):
        if q_position in {0, 1}:
            return k_position == 2
        elif q_position in {2, 3}:
            return k_position == 4
        elif q_position in {4}:
            return k_position == 5
        elif q_position in {5}:
            return k_position == 6
        elif q_position in {6}:
            return k_position == 1
        elif q_position in {7}:
            return k_position == 3

    attn_0_3_pattern = select_closest(positions, positions, predicate_0_3)
    attn_0_3_outputs = aggregate(attn_0_3_pattern, tokens)
    attn_0_3_output_scores = classifier_weights.loc[
        [("attn_0_3_outputs", str(v)) for v in attn_0_3_outputs]
    ]

    # num_attn_0_0 ####################################################
    def num_predicate_0_0(position, token):
        if position in {0, 2, 3}:
            return token == "1"
        elif position in {1, 7}:
            return token == "2"
        elif position in {4, 5, 6}:
            return token == "0"

    num_attn_0_0_pattern = select(tokens, positions, num_predicate_0_0)
    num_attn_0_0_outputs = aggregate_sum(num_attn_0_0_pattern, ones)
    num_attn_0_0_output_scores = classifier_weights.loc[
        [("num_attn_0_0_outputs", "_") for v in num_attn_0_0_outputs]
    ].mul(num_attn_0_0_outputs, axis=0)

    # num_attn_0_1 ####################################################
    def num_predicate_0_1(position, token):
        if position in {0, 6, 7}:
            return token == "0"
        elif position in {1}:
            return token == "4"
        elif position in {2, 3, 4}:
            return token == "2"
        elif position in {5}:
            return token == "1"

    num_attn_0_1_pattern = select(tokens, positions, num_predicate_0_1)
    num_attn_0_1_outputs = aggregate_sum(num_attn_0_1_pattern, ones)
    num_attn_0_1_output_scores = classifier_weights.loc[
        [("num_attn_0_1_outputs", "_") for v in num_attn_0_1_outputs]
    ].mul(num_attn_0_1_outputs, axis=0)

    # num_attn_0_2 ####################################################
    def num_predicate_0_2(position, token):
        if position in {0, 6, 7}:
            return token == "1"
        elif position in {1, 2}:
            return token == "0"
        elif position in {3}:
            return token == "<s>"
        elif position in {4, 5}:
            return token == "2"

    num_attn_0_2_pattern = select(tokens, positions, num_predicate_0_2)
    num_attn_0_2_outputs = aggregate_sum(num_attn_0_2_pattern, ones)
    num_attn_0_2_output_scores = classifier_weights.loc[
        [("num_attn_0_2_outputs", "_") for v in num_attn_0_2_outputs]
    ].mul(num_attn_0_2_outputs, axis=0)

    # num_attn_0_3 ####################################################
    def num_predicate_0_3(position, token):
        if position in {0}:
            return token == "3"
        elif position in {1, 7}:
            return token == "1"
        elif position in {2, 3}:
            return token == "0"
        elif position in {4}:
            return token == "<s>"
        elif position in {5, 6}:
            return token == "4"

    num_attn_0_3_pattern = select(tokens, positions, num_predicate_0_3)
    num_attn_0_3_outputs = aggregate_sum(num_attn_0_3_pattern, ones)
    num_attn_0_3_output_scores = classifier_weights.loc[
        [("num_attn_0_3_outputs", "_") for v in num_attn_0_3_outputs]
    ].mul(num_attn_0_3_outputs, axis=0)

    # mlp_0_0 #####################################################
    def mlp_0_0(position):
        key = position
        if key in {0, 2}:
            return 2
        elif key in {3}:
            return 0
        elif key in {7}:
            return 1
        elif key in {1}:
            return 3
        return 7

    mlp_0_0_outputs = [mlp_0_0(k0) for k0 in positions]
    mlp_0_0_output_scores = classifier_weights.loc[
        [("mlp_0_0_outputs", str(v)) for v in mlp_0_0_outputs]
    ]

    # mlp_0_1 #####################################################
    def mlp_0_1(position, attn_0_3_output):
        key = (position, attn_0_3_output)
        if key in {
            (0, "3"),
            (2, "3"),
            (2, "4"),
            (2, "</s>"),
            (2, "<s>"),
            (3, "3"),
            (4, "2"),
            (4, "3"),
            (4, "4"),
            (4, "<s>"),
            (5, "2"),
            (5, "3"),
            (5, "4"),
            (5, "<s>"),
            (7, "3"),
        }:
            return 7
        elif key in {
            (4, "</s>"),
            (5, "</s>"),
            (6, "0"),
            (6, "1"),
            (6, "2"),
            (6, "3"),
            (6, "4"),
            (6, "</s>"),
            (6, "<s>"),
        }:
            return 6
        elif key in {(0, "2"), (0, "4"), (0, "</s>"), (0, "<s>"), (7, "</s>")}:
            return 3
        elif key in {(1, "0"), (1, "1"), (1, "2"), (1, "3"), (1, "4"), (1, "<s>")}:
            return 5
        return 1

    mlp_0_1_outputs = [mlp_0_1(k0, k1) for k0, k1 in zip(positions, attn_0_3_outputs)]
    mlp_0_1_output_scores = classifier_weights.loc[
        [("mlp_0_1_outputs", str(v)) for v in mlp_0_1_outputs]
    ]

    # num_mlp_0_0 #################################################
    def num_mlp_0_0(num_attn_0_0_output, num_attn_0_2_output):
        key = (num_attn_0_0_output, num_attn_0_2_output)
        if key in {(0, 0), (0, 1)}:
            return 7
        return 2

    num_mlp_0_0_outputs = [
        num_mlp_0_0(k0, k1)
        for k0, k1 in zip(num_attn_0_0_outputs, num_attn_0_2_outputs)
    ]
    num_mlp_0_0_output_scores = classifier_weights.loc[
        [("num_mlp_0_0_outputs", str(v)) for v in num_mlp_0_0_outputs]
    ]

    # num_mlp_0_1 #################################################
    def num_mlp_0_1(num_attn_0_3_output):
        key = num_attn_0_3_output
        if key in {0, 1}:
            return 3
        return 1

    num_mlp_0_1_outputs = [num_mlp_0_1(k0) for k0 in num_attn_0_3_outputs]
    num_mlp_0_1_output_scores = classifier_weights.loc[
        [("num_mlp_0_1_outputs", str(v)) for v in num_mlp_0_1_outputs]
    ]

    # attn_1_0 ####################################################
    def predicate_1_0(position, token):
        if position in {0, 5}:
            return token == "2"
        elif position in {1}:
            return token == "0"
        elif position in {2}:
            return token == "3"
        elif position in {3, 4, 6}:
            return token == "4"
        elif position in {7}:
            return token == "<pad>"

    attn_1_0_pattern = select_closest(tokens, positions, predicate_1_0)
    attn_1_0_outputs = aggregate(attn_1_0_pattern, tokens)
    attn_1_0_output_scores = classifier_weights.loc[
        [("attn_1_0_outputs", str(v)) for v in attn_1_0_outputs]
    ]

    # attn_1_1 ####################################################
    def predicate_1_1(q_position, k_position):
        if q_position in {0}:
            return k_position == 2
        elif q_position in {1, 5, 6}:
            return k_position == 3
        elif q_position in {2}:
            return k_position == 1
        elif q_position in {3}:
            return k_position == 5
        elif q_position in {4, 7}:
            return k_position == 6

    attn_1_1_pattern = select_closest(positions, positions, predicate_1_1)
    attn_1_1_outputs = aggregate(attn_1_1_pattern, tokens)
    attn_1_1_output_scores = classifier_weights.loc[
        [("attn_1_1_outputs", str(v)) for v in attn_1_1_outputs]
    ]

    # attn_1_2 ####################################################
    def predicate_1_2(q_position, k_position):
        if q_position in {0}:
            return k_position == 4
        elif q_position in {1}:
            return k_position == 2
        elif q_position in {2, 5, 6}:
            return k_position == 3
        elif q_position in {3}:
            return k_position == 1
        elif q_position in {4}:
            return k_position == 6
        elif q_position in {7}:
            return k_position == 5

    attn_1_2_pattern = select_closest(positions, positions, predicate_1_2)
    attn_1_2_outputs = aggregate(attn_1_2_pattern, attn_0_1_outputs)
    attn_1_2_output_scores = classifier_weights.loc[
        [("attn_1_2_outputs", str(v)) for v in attn_1_2_outputs]
    ]

    # attn_1_3 ####################################################
    def predicate_1_3(attn_0_2_output, token):
        if attn_0_2_output in {"0"}:
            return token == "1"
        elif attn_0_2_output in {"1"}:
            return token == "0"
        elif attn_0_2_output in {"4", "2"}:
            return token == "<s>"
        elif attn_0_2_output in {"<s>", "3"}:
            return token == "</s>"
        elif attn_0_2_output in {"</s>"}:
            return token == "3"

    attn_1_3_pattern = select_closest(tokens, attn_0_2_outputs, predicate_1_3)
    attn_1_3_outputs = aggregate(attn_1_3_pattern, attn_0_1_outputs)
    attn_1_3_output_scores = classifier_weights.loc[
        [("attn_1_3_outputs", str(v)) for v in attn_1_3_outputs]
    ]

    # num_attn_1_0 ####################################################
    def num_predicate_1_0(position, token):
        if position in {0, 4, 5, 6, 7}:
            return token == "3"
        elif position in {1, 2}:
            return token == "0"
        elif position in {3}:
            return token == "</s>"

    num_attn_1_0_pattern = select(tokens, positions, num_predicate_1_0)
    num_attn_1_0_outputs = aggregate_sum(num_attn_1_0_pattern, ones)
    num_attn_1_0_output_scores = classifier_weights.loc[
        [("num_attn_1_0_outputs", "_") for v in num_attn_1_0_outputs]
    ].mul(num_attn_1_0_outputs, axis=0)

    # num_attn_1_1 ####################################################
    def num_predicate_1_1(position, attn_0_3_output):
        if position in {0, 1, 2}:
            return attn_0_3_output == "2"
        elif position in {3, 4, 5, 6}:
            return attn_0_3_output == "1"
        elif position in {7}:
            return attn_0_3_output == "0"

    num_attn_1_1_pattern = select(attn_0_3_outputs, positions, num_predicate_1_1)
    num_attn_1_1_outputs = aggregate_sum(num_attn_1_1_pattern, num_attn_0_2_outputs)
    num_attn_1_1_output_scores = classifier_weights.loc[
        [("num_attn_1_1_outputs", "_") for v in num_attn_1_1_outputs]
    ].mul(num_attn_1_1_outputs, axis=0)

    # num_attn_1_2 ####################################################
    def num_predicate_1_2(q_mlp_0_0_output, k_mlp_0_0_output):
        if q_mlp_0_0_output in {0, 1, 2, 3, 4, 5, 7}:
            return k_mlp_0_0_output == 2
        elif q_mlp_0_0_output in {6}:
            return k_mlp_0_0_output == 3

    num_attn_1_2_pattern = select(mlp_0_0_outputs, mlp_0_0_outputs, num_predicate_1_2)
    num_attn_1_2_outputs = aggregate_sum(num_attn_1_2_pattern, num_attn_0_2_outputs)
    num_attn_1_2_output_scores = classifier_weights.loc[
        [("num_attn_1_2_outputs", "_") for v in num_attn_1_2_outputs]
    ].mul(num_attn_1_2_outputs, axis=0)

    # num_attn_1_3 ####################################################
    def num_predicate_1_3(num_mlp_0_1_output, mlp_0_0_output):
        if num_mlp_0_1_output in {0, 1, 2, 3, 4, 6}:
            return mlp_0_0_output == 2
        elif num_mlp_0_1_output in {5, 7}:
            return mlp_0_0_output == 1

    num_attn_1_3_pattern = select(
        mlp_0_0_outputs, num_mlp_0_1_outputs, num_predicate_1_3
    )
    num_attn_1_3_outputs = aggregate_sum(num_attn_1_3_pattern, num_attn_0_2_outputs)
    num_attn_1_3_output_scores = classifier_weights.loc[
        [("num_attn_1_3_outputs", "_") for v in num_attn_1_3_outputs]
    ].mul(num_attn_1_3_outputs, axis=0)

    # mlp_1_0 #####################################################
    def mlp_1_0(attn_1_3_output, position):
        key = (attn_1_3_output, position)
        if key in {
            ("0", 1),
            ("0", 2),
            ("0", 3),
            ("1", 0),
            ("1", 1),
            ("1", 2),
            ("1", 3),
            ("1", 7),
            ("2", 0),
            ("2", 1),
            ("2", 2),
            ("2", 3),
            ("2", 7),
            ("3", 0),
            ("3", 1),
            ("3", 2),
            ("3", 3),
            ("3", 7),
            ("4", 1),
            ("4", 2),
            ("4", 3),
            ("</s>", 1),
            ("</s>", 2),
            ("</s>", 3),
            ("<s>", 1),
            ("<s>", 2),
            ("<s>", 3),
        }:
            return 4
        return 0

    mlp_1_0_outputs = [mlp_1_0(k0, k1) for k0, k1 in zip(attn_1_3_outputs, positions)]
    mlp_1_0_output_scores = classifier_weights.loc[
        [("mlp_1_0_outputs", str(v)) for v in mlp_1_0_outputs]
    ]

    # mlp_1_1 #####################################################
    def mlp_1_1(position, attn_1_3_output):
        key = (position, attn_1_3_output)
        if key in {
            (0, "0"),
            (0, "1"),
            (1, "0"),
            (1, "1"),
            (1, "2"),
            (1, "3"),
            (1, "4"),
            (1, "</s>"),
            (1, "<s>"),
            (2, "0"),
            (2, "1"),
            (2, "2"),
            (2, "3"),
            (2, "4"),
            (2, "<s>"),
            (7, "0"),
            (7, "1"),
        }:
            return 0
        elif key in {
            (0, "<s>"),
            (4, "4"),
            (4, "</s>"),
            (4, "<s>"),
            (6, "0"),
            (6, "1"),
            (6, "2"),
            (6, "3"),
            (6, "4"),
            (6, "</s>"),
            (6, "<s>"),
            (7, "</s>"),
            (7, "<s>"),
        }:
            return 1
        elif key in {(2, "</s>")}:
            return 3
        return 5

    mlp_1_1_outputs = [mlp_1_1(k0, k1) for k0, k1 in zip(positions, attn_1_3_outputs)]
    mlp_1_1_output_scores = classifier_weights.loc[
        [("mlp_1_1_outputs", str(v)) for v in mlp_1_1_outputs]
    ]

    # num_mlp_1_0 #################################################
    def num_mlp_1_0(num_attn_1_2_output, num_attn_0_2_output):
        key = (num_attn_1_2_output, num_attn_0_2_output)
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
            (1, 0),
            (1, 1),
            (1, 2),
            (1, 3),
            (1, 4),
            (1, 5),
            (2, 0),
            (2, 1),
        }:
            return 7
        return 5

    num_mlp_1_0_outputs = [
        num_mlp_1_0(k0, k1)
        for k0, k1 in zip(num_attn_1_2_outputs, num_attn_0_2_outputs)
    ]
    num_mlp_1_0_output_scores = classifier_weights.loc[
        [("num_mlp_1_0_outputs", str(v)) for v in num_mlp_1_0_outputs]
    ]

    # num_mlp_1_1 #################################################
    def num_mlp_1_1(num_attn_1_3_output, num_attn_0_2_output):
        key = (num_attn_1_3_output, num_attn_0_2_output)
        if key in {(0, 0), (1, 0)}:
            return 6
        return 2

    num_mlp_1_1_outputs = [
        num_mlp_1_1(k0, k1)
        for k0, k1 in zip(num_attn_1_3_outputs, num_attn_0_2_outputs)
    ]
    num_mlp_1_1_output_scores = classifier_weights.loc[
        [("num_mlp_1_1_outputs", str(v)) for v in num_mlp_1_1_outputs]
    ]

    # attn_2_0 ####################################################
    def predicate_2_0(attn_0_3_output, token):
        if attn_0_3_output in {"4", "1", "0", "2"}:
            return token == "3"
        elif attn_0_3_output in {"3"}:
            return token == "2"
        elif attn_0_3_output in {"</s>"}:
            return token == "4"
        elif attn_0_3_output in {"<s>"}:
            return token == "<pad>"

    attn_2_0_pattern = select_closest(tokens, attn_0_3_outputs, predicate_2_0)
    attn_2_0_outputs = aggregate(attn_2_0_pattern, tokens)
    attn_2_0_output_scores = classifier_weights.loc[
        [("attn_2_0_outputs", str(v)) for v in attn_2_0_outputs]
    ]

    # attn_2_1 ####################################################
    def predicate_2_1(position, token):
        if position in {0, 7}:
            return token == "2"
        elif position in {1, 2}:
            return token == "0"
        elif position in {3}:
            return token == "3"
        elif position in {4, 5, 6}:
            return token == "4"

    attn_2_1_pattern = select_closest(tokens, positions, predicate_2_1)
    attn_2_1_outputs = aggregate(attn_2_1_pattern, tokens)
    attn_2_1_output_scores = classifier_weights.loc[
        [("attn_2_1_outputs", str(v)) for v in attn_2_1_outputs]
    ]

    # attn_2_2 ####################################################
    def predicate_2_2(mlp_0_0_output, token):
        if mlp_0_0_output in {0, 2}:
            return token == "</s>"
        elif mlp_0_0_output in {1, 5}:
            return token == "0"
        elif mlp_0_0_output in {3}:
            return token == "1"
        elif mlp_0_0_output in {4}:
            return token == "2"
        elif mlp_0_0_output in {6, 7}:
            return token == "4"

    attn_2_2_pattern = select_closest(tokens, mlp_0_0_outputs, predicate_2_2)
    attn_2_2_outputs = aggregate(attn_2_2_pattern, attn_1_1_outputs)
    attn_2_2_output_scores = classifier_weights.loc[
        [("attn_2_2_outputs", str(v)) for v in attn_2_2_outputs]
    ]

    # attn_2_3 ####################################################
    def predicate_2_3(attn_1_0_output, position):
        if attn_1_0_output in {"1", "0"}:
            return position == 1
        elif attn_1_0_output in {"2"}:
            return position == 2
        elif attn_1_0_output in {"3"}:
            return position == 6
        elif attn_1_0_output in {"4"}:
            return position == 4
        elif attn_1_0_output in {"</s>"}:
            return position == 5
        elif attn_1_0_output in {"<s>"}:
            return position == 0

    attn_2_3_pattern = select_closest(positions, attn_1_0_outputs, predicate_2_3)
    attn_2_3_outputs = aggregate(attn_2_3_pattern, attn_1_2_outputs)
    attn_2_3_output_scores = classifier_weights.loc[
        [("attn_2_3_outputs", str(v)) for v in attn_2_3_outputs]
    ]

    # num_attn_2_0 ####################################################
    def num_predicate_2_0(mlp_1_0_output, token):
        if mlp_1_0_output in {0, 1, 2, 4, 5, 6, 7}:
            return token == "2"
        elif mlp_1_0_output in {3}:
            return token == "</s>"

    num_attn_2_0_pattern = select(tokens, mlp_1_0_outputs, num_predicate_2_0)
    num_attn_2_0_outputs = aggregate_sum(num_attn_2_0_pattern, ones)
    num_attn_2_0_output_scores = classifier_weights.loc[
        [("num_attn_2_0_outputs", "_") for v in num_attn_2_0_outputs]
    ].mul(num_attn_2_0_outputs, axis=0)

    # num_attn_2_1 ####################################################
    def num_predicate_2_1(position, num_mlp_0_0_output):
        if position in {0, 5, 7}:
            return num_mlp_0_0_output == 1
        elif position in {1, 2, 3, 4}:
            return num_mlp_0_0_output == 2
        elif position in {6}:
            return num_mlp_0_0_output == 0

    num_attn_2_1_pattern = select(num_mlp_0_0_outputs, positions, num_predicate_2_1)
    num_attn_2_1_outputs = aggregate_sum(num_attn_2_1_pattern, num_attn_0_0_outputs)
    num_attn_2_1_output_scores = classifier_weights.loc[
        [("num_attn_2_1_outputs", "_") for v in num_attn_2_1_outputs]
    ].mul(num_attn_2_1_outputs, axis=0)

    # num_attn_2_2 ####################################################
    def num_predicate_2_2(mlp_0_1_output, token):
        if mlp_0_1_output in {0, 1, 2, 3, 5, 7}:
            return token == "3"
        elif mlp_0_1_output in {4, 6}:
            return token == "2"

    num_attn_2_2_pattern = select(tokens, mlp_0_1_outputs, num_predicate_2_2)
    num_attn_2_2_outputs = aggregate_sum(num_attn_2_2_pattern, num_attn_1_2_outputs)
    num_attn_2_2_output_scores = classifier_weights.loc[
        [("num_attn_2_2_outputs", "_") for v in num_attn_2_2_outputs]
    ].mul(num_attn_2_2_outputs, axis=0)

    # num_attn_2_3 ####################################################
    def num_predicate_2_3(position, token):
        if position in {0, 1, 2, 3, 4, 5}:
            return token == "0"
        elif position in {6}:
            return token == "<pad>"
        elif position in {7}:
            return token == "1"

    num_attn_2_3_pattern = select(tokens, positions, num_predicate_2_3)
    num_attn_2_3_outputs = aggregate_sum(num_attn_2_3_pattern, num_attn_1_2_outputs)
    num_attn_2_3_output_scores = classifier_weights.loc[
        [("num_attn_2_3_outputs", "_") for v in num_attn_2_3_outputs]
    ].mul(num_attn_2_3_outputs, axis=0)

    # mlp_2_0 #####################################################
    def mlp_2_0(position, attn_2_3_output):
        key = (position, attn_2_3_output)
        if key in {
            (0, "</s>"),
            (0, "<s>"),
            (3, "0"),
            (3, "</s>"),
            (3, "<s>"),
            (5, "0"),
            (5, "1"),
            (5, "2"),
            (5, "3"),
            (5, "4"),
            (5, "</s>"),
            (5, "<s>"),
            (7, "0"),
            (7, "<s>"),
        }:
            return 0
        elif key in {
            (0, "0"),
            (1, "0"),
            (1, "1"),
            (1, "2"),
            (1, "3"),
            (1, "4"),
            (1, "</s>"),
            (1, "<s>"),
            (2, "0"),
        }:
            return 6
        elif key in {
            (0, "2"),
            (2, "1"),
            (2, "2"),
            (2, "3"),
            (2, "4"),
            (2, "</s>"),
            (2, "<s>"),
        }:
            return 4
        return 2

    mlp_2_0_outputs = [mlp_2_0(k0, k1) for k0, k1 in zip(positions, attn_2_3_outputs)]
    mlp_2_0_output_scores = classifier_weights.loc[
        [("mlp_2_0_outputs", str(v)) for v in mlp_2_0_outputs]
    ]

    # mlp_2_1 #####################################################
    def mlp_2_1(position, attn_2_2_output):
        key = (position, attn_2_2_output)
        if key in {
            (0, "0"),
            (1, "0"),
            (1, "1"),
            (1, "2"),
            (1, "3"),
            (1, "4"),
            (1, "</s>"),
            (1, "<s>"),
            (2, "0"),
            (2, "1"),
            (2, "2"),
            (2, "3"),
            (2, "4"),
            (2, "</s>"),
            (2, "<s>"),
        }:
            return 2
        return 7

    mlp_2_1_outputs = [mlp_2_1(k0, k1) for k0, k1 in zip(positions, attn_2_2_outputs)]
    mlp_2_1_output_scores = classifier_weights.loc[
        [("mlp_2_1_outputs", str(v)) for v in mlp_2_1_outputs]
    ]

    # num_mlp_2_0 #################################################
    def num_mlp_2_0(num_attn_0_3_output, num_attn_2_3_output):
        key = (num_attn_0_3_output, num_attn_2_3_output)
        if key in {
            (1, 0),
            (2, 0),
            (3, 0),
            (4, 0),
            (5, 0),
            (6, 0),
            (7, 0),
            (8, 0),
            (9, 0),
            (10, 0),
            (11, 0),
            (12, 0),
            (13, 0),
            (14, 0),
            (15, 0),
            (16, 0),
            (17, 0),
            (18, 0),
            (19, 0),
            (20, 0),
            (21, 0),
            (22, 0),
            (23, 0),
        }:
            return 6
        elif key in {(0, 1), (0, 2)}:
            return 0
        elif key in {(0, 0)}:
            return 4
        return 7

    num_mlp_2_0_outputs = [
        num_mlp_2_0(k0, k1)
        for k0, k1 in zip(num_attn_0_3_outputs, num_attn_2_3_outputs)
    ]
    num_mlp_2_0_output_scores = classifier_weights.loc[
        [("num_mlp_2_0_outputs", str(v)) for v in num_mlp_2_0_outputs]
    ]

    # num_mlp_2_1 #################################################
    def num_mlp_2_1(num_attn_2_3_output, num_attn_2_2_output):
        key = (num_attn_2_3_output, num_attn_2_2_output)
        if key in {(0, 0)}:
            return 3
        return 7

    num_mlp_2_1_outputs = [
        num_mlp_2_1(k0, k1)
        for k0, k1 in zip(num_attn_2_3_outputs, num_attn_2_2_outputs)
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


examples = [
    (
        ["<s>", "3", "1", "4", "2", "4", "0", "</s>"],
        ["<pad>", "0", "1", "2", "3", "4", "4", "<pad>"],
    ),
    (
        ["<s>", "4", "0", "4", "2", "0", "</s>"],
        ["<pad>", "0", "0", "2", "4", "4", "<pad>"],
    ),
    (
        ["<s>", "1", "0", "2", "4", "3", "2", "</s>"],
        ["<pad>", "0", "1", "2", "2", "3", "4", "<pad>"],
    ),
    (
        ["<s>", "3", "3", "2", "4", "4", "4", "</s>"],
        ["<pad>", "2", "3", "3", "4", "4", "4", "<pad>"],
    ),
    (
        ["<s>", "3", "3", "2", "2", "2", "0", "</s>"],
        ["<pad>", "0", "2", "2", "2", "3", "3", "<pad>"],
    ),
    (["<s>", "4", "4", "0", "</s>"], ["<pad>", "0", "4", "4", "<pad>"]),
    (
        ["<s>", "3", "2", "0", "1", "0", "0", "</s>"],
        ["<pad>", "0", "0", "0", "1", "2", "3", "<pad>"],
    ),
    (
        ["<s>", "2", "3", "2", "1", "1", "0", "</s>"],
        ["<pad>", "0", "1", "1", "2", "2", "3", "<pad>"],
    ),
    (
        ["<s>", "0", "0", "3", "3", "4", "</s>"],
        ["<pad>", "0", "0", "3", "3", "4", "<pad>"],
    ),
    (
        ["<s>", "4", "4", "0", "2", "4", "4", "</s>"],
        ["<pad>", "0", "2", "4", "4", "4", "4", "<pad>"],
    ),
]
for x, y in examples:
    print(f"x: {x}")
    print(f"y: {y}")
    y_hat = run(x)
    print(f"y_hat: {y_hat}")
    print()
