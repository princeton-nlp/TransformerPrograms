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
        "programs/rasp/double_hist/double_hist_weights.csv",
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
            return k_position == 4
        elif q_position in {1, 2, 3, 4, 5, 6, 7}:
            return k_position == 6

    attn_0_0_pattern = select_closest(positions, positions, predicate_0_0)
    attn_0_0_outputs = aggregate(attn_0_0_pattern, positions)
    attn_0_0_output_scores = classifier_weights.loc[
        [("attn_0_0_outputs", str(v)) for v in attn_0_0_outputs]
    ]

    # attn_0_1 ####################################################
    def predicate_0_1(q_position, k_position):
        if q_position in {0, 2, 4, 5, 7}:
            return k_position == 7
        elif q_position in {1, 3, 6}:
            return k_position == 6

    attn_0_1_pattern = select_closest(positions, positions, predicate_0_1)
    attn_0_1_outputs = aggregate(attn_0_1_pattern, positions)
    attn_0_1_output_scores = classifier_weights.loc[
        [("attn_0_1_outputs", str(v)) for v in attn_0_1_outputs]
    ]

    # attn_0_2 ####################################################
    def predicate_0_2(q_position, k_position):
        if q_position in {0}:
            return k_position == 4
        elif q_position in {1, 2, 3, 4, 5, 6, 7}:
            return k_position == 6

    attn_0_2_pattern = select_closest(positions, positions, predicate_0_2)
    attn_0_2_outputs = aggregate(attn_0_2_pattern, positions)
    attn_0_2_output_scores = classifier_weights.loc[
        [("attn_0_2_outputs", str(v)) for v in attn_0_2_outputs]
    ]

    # attn_0_3 ####################################################
    def predicate_0_3(q_position, k_position):
        if q_position in {0}:
            return k_position == 4
        elif q_position in {1}:
            return k_position == 5
        elif q_position in {2, 3, 4, 6}:
            return k_position == 6
        elif q_position in {5}:
            return k_position == 2
        elif q_position in {7}:
            return k_position == 7

    attn_0_3_pattern = select_closest(positions, positions, predicate_0_3)
    attn_0_3_outputs = aggregate(attn_0_3_pattern, positions)
    attn_0_3_output_scores = classifier_weights.loc[
        [("attn_0_3_outputs", str(v)) for v in attn_0_3_outputs]
    ]

    # num_attn_0_0 ####################################################
    def num_predicate_0_0(q_token, k_token):
        if q_token in {"0"}:
            return k_token == "0"
        elif q_token in {"1"}:
            return k_token == "1"
        elif q_token in {"2"}:
            return k_token == "2"
        elif q_token in {"3"}:
            return k_token == "3"
        elif q_token in {"4"}:
            return k_token == "4"
        elif q_token in {"5"}:
            return k_token == "5"
        elif q_token in {"<s>"}:
            return k_token == "<pad>"

    num_attn_0_0_pattern = select(tokens, tokens, num_predicate_0_0)
    num_attn_0_0_outputs = aggregate_sum(num_attn_0_0_pattern, ones)
    num_attn_0_0_output_scores = classifier_weights.loc[
        [("num_attn_0_0_outputs", "_") for v in num_attn_0_0_outputs]
    ].mul(num_attn_0_0_outputs, axis=0)

    # num_attn_0_1 ####################################################
    def num_predicate_0_1(q_position, k_position):
        if q_position in {0}:
            return k_position == 7
        elif q_position in {1, 6}:
            return k_position == 3
        elif q_position in {2, 4}:
            return k_position == 4
        elif q_position in {3, 5}:
            return k_position == 5
        elif q_position in {7}:
            return k_position == 6

    num_attn_0_1_pattern = select(positions, positions, num_predicate_0_1)
    num_attn_0_1_outputs = aggregate_sum(num_attn_0_1_pattern, ones)
    num_attn_0_1_output_scores = classifier_weights.loc[
        [("num_attn_0_1_outputs", "_") for v in num_attn_0_1_outputs]
    ].mul(num_attn_0_1_outputs, axis=0)

    # num_attn_0_2 ####################################################
    def num_predicate_0_2(token, position):
        if token in {"1", "4", "0", "2", "3", "5"}:
            return position == 5
        elif token in {"<s>"}:
            return position == 7

    num_attn_0_2_pattern = select(positions, tokens, num_predicate_0_2)
    num_attn_0_2_outputs = aggregate_sum(num_attn_0_2_pattern, ones)
    num_attn_0_2_output_scores = classifier_weights.loc[
        [("num_attn_0_2_outputs", "_") for v in num_attn_0_2_outputs]
    ].mul(num_attn_0_2_outputs, axis=0)

    # num_attn_0_3 ####################################################
    def num_predicate_0_3(q_position, k_position):
        if q_position in {0, 1, 4}:
            return k_position == 2
        elif q_position in {2, 3, 5, 6}:
            return k_position == 4
        elif q_position in {7}:
            return k_position == 0

    num_attn_0_3_pattern = select(positions, positions, num_predicate_0_3)
    num_attn_0_3_outputs = aggregate_sum(num_attn_0_3_pattern, ones)
    num_attn_0_3_output_scores = classifier_weights.loc[
        [("num_attn_0_3_outputs", "_") for v in num_attn_0_3_outputs]
    ].mul(num_attn_0_3_outputs, axis=0)

    # mlp_0_0 #####################################################
    def mlp_0_0(attn_0_3_output, attn_0_2_output):
        key = (attn_0_3_output, attn_0_2_output)
        if key in {(3, 1), (3, 2), (3, 3), (3, 5), (3, 6), (3, 7)}:
            return 3
        return 6

    mlp_0_0_outputs = [
        mlp_0_0(k0, k1) for k0, k1 in zip(attn_0_3_outputs, attn_0_2_outputs)
    ]
    mlp_0_0_output_scores = classifier_weights.loc[
        [("mlp_0_0_outputs", str(v)) for v in mlp_0_0_outputs]
    ]

    # num_mlp_0_0 #################################################
    def num_mlp_0_0(num_attn_0_0_output):
        key = num_attn_0_0_output
        if key in {0, 1}:
            return 3
        return 4

    num_mlp_0_0_outputs = [num_mlp_0_0(k0) for k0 in num_attn_0_0_outputs]
    num_mlp_0_0_output_scores = classifier_weights.loc[
        [("num_mlp_0_0_outputs", str(v)) for v in num_mlp_0_0_outputs]
    ]

    # attn_1_0 ####################################################
    def predicate_1_0(q_position, k_position):
        if q_position in {0, 4, 6}:
            return k_position == 3
        elif q_position in {1, 2}:
            return k_position == 5
        elif q_position in {3}:
            return k_position == 2
        elif q_position in {5}:
            return k_position == 1
        elif q_position in {7}:
            return k_position == 6

    attn_1_0_pattern = select_closest(positions, positions, predicate_1_0)
    attn_1_0_outputs = aggregate(attn_1_0_pattern, num_mlp_0_0_outputs)
    attn_1_0_output_scores = classifier_weights.loc[
        [("attn_1_0_outputs", str(v)) for v in attn_1_0_outputs]
    ]

    # attn_1_1 ####################################################
    def predicate_1_1(num_mlp_0_0_output, position):
        if num_mlp_0_0_output in {0, 1, 2, 4}:
            return position == 6
        elif num_mlp_0_0_output in {3, 6, 7}:
            return position == 7
        elif num_mlp_0_0_output in {5}:
            return position == 3

    attn_1_1_pattern = select_closest(positions, num_mlp_0_0_outputs, predicate_1_1)
    attn_1_1_outputs = aggregate(attn_1_1_pattern, attn_0_3_outputs)
    attn_1_1_output_scores = classifier_weights.loc[
        [("attn_1_1_outputs", str(v)) for v in attn_1_1_outputs]
    ]

    # attn_1_2 ####################################################
    def predicate_1_2(q_position, k_position):
        if q_position in {0, 5, 7}:
            return k_position == 6
        elif q_position in {1, 2, 4}:
            return k_position == 3
        elif q_position in {3}:
            return k_position == 2
        elif q_position in {6}:
            return k_position == 5

    attn_1_2_pattern = select_closest(positions, positions, predicate_1_2)
    attn_1_2_outputs = aggregate(attn_1_2_pattern, num_mlp_0_0_outputs)
    attn_1_2_output_scores = classifier_weights.loc[
        [("attn_1_2_outputs", str(v)) for v in attn_1_2_outputs]
    ]

    # attn_1_3 ####################################################
    def predicate_1_3(attn_0_1_output, position):
        if attn_0_1_output in {0, 1}:
            return position == 5
        elif attn_0_1_output in {2, 4}:
            return position == 7
        elif attn_0_1_output in {3}:
            return position == 6
        elif attn_0_1_output in {5}:
            return position == 0
        elif attn_0_1_output in {6}:
            return position == 2
        elif attn_0_1_output in {7}:
            return position == 4

    attn_1_3_pattern = select_closest(positions, attn_0_1_outputs, predicate_1_3)
    attn_1_3_outputs = aggregate(attn_1_3_pattern, attn_0_3_outputs)
    attn_1_3_output_scores = classifier_weights.loc[
        [("attn_1_3_outputs", str(v)) for v in attn_1_3_outputs]
    ]

    # num_attn_1_0 ####################################################
    def num_predicate_1_0(attn_0_2_output, position):
        if attn_0_2_output in {0, 1, 2, 3, 4}:
            return position == 5
        elif attn_0_2_output in {5, 6}:
            return position == 7
        elif attn_0_2_output in {7}:
            return position == 3

    num_attn_1_0_pattern = select(positions, attn_0_2_outputs, num_predicate_1_0)
    num_attn_1_0_outputs = aggregate_sum(num_attn_1_0_pattern, num_attn_0_3_outputs)
    num_attn_1_0_output_scores = classifier_weights.loc[
        [("num_attn_1_0_outputs", "_") for v in num_attn_1_0_outputs]
    ].mul(num_attn_1_0_outputs, axis=0)

    # num_attn_1_1 ####################################################
    def num_predicate_1_1(attn_0_0_output, num_mlp_0_0_output):
        if attn_0_0_output in {0, 1, 2, 5, 6}:
            return num_mlp_0_0_output == 3
        elif attn_0_0_output in {3, 4}:
            return num_mlp_0_0_output == 6
        elif attn_0_0_output in {7}:
            return num_mlp_0_0_output == 7

    num_attn_1_1_pattern = select(
        num_mlp_0_0_outputs, attn_0_0_outputs, num_predicate_1_1
    )
    num_attn_1_1_outputs = aggregate_sum(num_attn_1_1_pattern, num_attn_0_1_outputs)
    num_attn_1_1_output_scores = classifier_weights.loc[
        [("num_attn_1_1_outputs", "_") for v in num_attn_1_1_outputs]
    ].mul(num_attn_1_1_outputs, axis=0)

    # num_attn_1_2 ####################################################
    def num_predicate_1_2(attn_0_0_output, attn_0_1_output):
        if attn_0_0_output in {0, 1, 2, 3}:
            return attn_0_1_output == 6
        elif attn_0_0_output in {4, 5}:
            return attn_0_1_output == 7
        elif attn_0_0_output in {6}:
            return attn_0_1_output == 4
        elif attn_0_0_output in {7}:
            return attn_0_1_output == 2

    num_attn_1_2_pattern = select(attn_0_1_outputs, attn_0_0_outputs, num_predicate_1_2)
    num_attn_1_2_outputs = aggregate_sum(num_attn_1_2_pattern, num_attn_0_2_outputs)
    num_attn_1_2_output_scores = classifier_weights.loc[
        [("num_attn_1_2_outputs", "_") for v in num_attn_1_2_outputs]
    ].mul(num_attn_1_2_outputs, axis=0)

    # num_attn_1_3 ####################################################
    def num_predicate_1_3(q_num_mlp_0_0_output, k_num_mlp_0_0_output):
        if q_num_mlp_0_0_output in {0, 1, 2, 3, 5}:
            return k_num_mlp_0_0_output == 4
        elif q_num_mlp_0_0_output in {4}:
            return k_num_mlp_0_0_output == 3
        elif q_num_mlp_0_0_output in {6}:
            return k_num_mlp_0_0_output == 2
        elif q_num_mlp_0_0_output in {7}:
            return k_num_mlp_0_0_output == 1

    num_attn_1_3_pattern = select(
        num_mlp_0_0_outputs, num_mlp_0_0_outputs, num_predicate_1_3
    )
    num_attn_1_3_outputs = aggregate_sum(num_attn_1_3_pattern, num_attn_0_0_outputs)
    num_attn_1_3_output_scores = classifier_weights.loc[
        [("num_attn_1_3_outputs", "_") for v in num_attn_1_3_outputs]
    ].mul(num_attn_1_3_outputs, axis=0)

    # mlp_1_0 #####################################################
    def mlp_1_0(attn_1_2_output, num_mlp_0_0_output):
        key = (attn_1_2_output, num_mlp_0_0_output)
        if key in {
            (0, 4),
            (1, 4),
            (1, 6),
            (2, 2),
            (2, 4),
            (2, 6),
            (3, 4),
            (4, 2),
            (4, 4),
            (4, 5),
            (4, 6),
            (5, 4),
            (5, 6),
            (6, 4),
            (6, 6),
            (7, 4),
        }:
            return 3
        elif key in {
            (2, 0),
            (3, 0),
            (3, 1),
            (3, 2),
            (3, 3),
            (3, 5),
            (3, 6),
            (3, 7),
            (4, 0),
            (6, 0),
        }:
            return 7
        return 1

    mlp_1_0_outputs = [
        mlp_1_0(k0, k1) for k0, k1 in zip(attn_1_2_outputs, num_mlp_0_0_outputs)
    ]
    mlp_1_0_output_scores = classifier_weights.loc[
        [("mlp_1_0_outputs", str(v)) for v in mlp_1_0_outputs]
    ]

    # num_mlp_1_0 #################################################
    def num_mlp_1_0(num_attn_0_0_output, num_attn_1_3_output):
        key = (num_attn_0_0_output, num_attn_1_3_output)
        if key in {(1, 0), (2, 0), (3, 0), (4, 0), (5, 0)}:
            return 7
        elif key in {(0, 0)}:
            return 6
        return 5

    num_mlp_1_0_outputs = [
        num_mlp_1_0(k0, k1)
        for k0, k1 in zip(num_attn_0_0_outputs, num_attn_1_3_outputs)
    ]
    num_mlp_1_0_output_scores = classifier_weights.loc[
        [("num_mlp_1_0_outputs", str(v)) for v in num_mlp_1_0_outputs]
    ]

    # attn_2_0 ####################################################
    def predicate_2_0(q_token, k_token):
        if q_token in {"4", "5", "0", "2"}:
            return k_token == "3"
        elif q_token in {"1"}:
            return k_token == "2"
        elif q_token in {"3"}:
            return k_token == "5"
        elif q_token in {"<s>"}:
            return k_token == "0"

    attn_2_0_pattern = select_closest(tokens, tokens, predicate_2_0)
    attn_2_0_outputs = aggregate(attn_2_0_pattern, attn_0_1_outputs)
    attn_2_0_output_scores = classifier_weights.loc[
        [("attn_2_0_outputs", str(v)) for v in attn_2_0_outputs]
    ]

    # attn_2_1 ####################################################
    def predicate_2_1(num_mlp_1_0_output, position):
        if num_mlp_1_0_output in {0, 7}:
            return position == 6
        elif num_mlp_1_0_output in {1, 2, 3, 4, 5, 6}:
            return position == 7

    attn_2_1_pattern = select_closest(positions, num_mlp_1_0_outputs, predicate_2_1)
    attn_2_1_outputs = aggregate(attn_2_1_pattern, positions)
    attn_2_1_output_scores = classifier_weights.loc[
        [("attn_2_1_outputs", str(v)) for v in attn_2_1_outputs]
    ]

    # attn_2_2 ####################################################
    def predicate_2_2(attn_1_3_output, num_mlp_0_0_output):
        if attn_1_3_output in {0, 2, 3, 4, 6}:
            return num_mlp_0_0_output == 4
        elif attn_1_3_output in {1, 7}:
            return num_mlp_0_0_output == 5
        elif attn_1_3_output in {5}:
            return num_mlp_0_0_output == 3

    attn_2_2_pattern = select_closest(
        num_mlp_0_0_outputs, attn_1_3_outputs, predicate_2_2
    )
    attn_2_2_outputs = aggregate(attn_2_2_pattern, attn_1_0_outputs)
    attn_2_2_output_scores = classifier_weights.loc[
        [("attn_2_2_outputs", str(v)) for v in attn_2_2_outputs]
    ]

    # attn_2_3 ####################################################
    def predicate_2_3(q_attn_0_0_output, k_attn_0_0_output):
        if q_attn_0_0_output in {0, 1, 2, 4, 7}:
            return k_attn_0_0_output == 6
        elif q_attn_0_0_output in {3, 6}:
            return k_attn_0_0_output == 4
        elif q_attn_0_0_output in {5}:
            return k_attn_0_0_output == 2

    attn_2_3_pattern = select_closest(attn_0_0_outputs, attn_0_0_outputs, predicate_2_3)
    attn_2_3_outputs = aggregate(attn_2_3_pattern, attn_1_3_outputs)
    attn_2_3_output_scores = classifier_weights.loc[
        [("attn_2_3_outputs", str(v)) for v in attn_2_3_outputs]
    ]

    # num_attn_2_0 ####################################################
    def num_predicate_2_0(mlp_1_0_output, num_mlp_0_0_output):
        if mlp_1_0_output in {0, 1, 2, 4, 5, 6, 7}:
            return num_mlp_0_0_output == 3
        elif mlp_1_0_output in {3}:
            return num_mlp_0_0_output == 5

    num_attn_2_0_pattern = select(
        num_mlp_0_0_outputs, mlp_1_0_outputs, num_predicate_2_0
    )
    num_attn_2_0_outputs = aggregate_sum(num_attn_2_0_pattern, num_attn_1_1_outputs)
    num_attn_2_0_output_scores = classifier_weights.loc[
        [("num_attn_2_0_outputs", "_") for v in num_attn_2_0_outputs]
    ].mul(num_attn_2_0_outputs, axis=0)

    # num_attn_2_1 ####################################################
    def num_predicate_2_1(q_token, k_token):
        if q_token in {"0"}:
            return k_token == "0"
        elif q_token in {"1"}:
            return k_token == "1"
        elif q_token in {"2"}:
            return k_token == "2"
        elif q_token in {"3"}:
            return k_token == "3"
        elif q_token in {"4"}:
            return k_token == "4"
        elif q_token in {"5"}:
            return k_token == "5"
        elif q_token in {"<s>"}:
            return k_token == "<s>"

    num_attn_2_1_pattern = select(tokens, tokens, num_predicate_2_1)
    num_attn_2_1_outputs = aggregate_sum(num_attn_2_1_pattern, num_attn_0_0_outputs)
    num_attn_2_1_output_scores = classifier_weights.loc[
        [("num_attn_2_1_outputs", "_") for v in num_attn_2_1_outputs]
    ].mul(num_attn_2_1_outputs, axis=0)

    # num_attn_2_2 ####################################################
    def num_predicate_2_2(attn_1_3_output, num_mlp_0_0_output):
        if attn_1_3_output in {0, 1, 2, 3, 4, 5, 6, 7}:
            return num_mlp_0_0_output == 4

    num_attn_2_2_pattern = select(
        num_mlp_0_0_outputs, attn_1_3_outputs, num_predicate_2_2
    )
    num_attn_2_2_outputs = aggregate_sum(num_attn_2_2_pattern, num_attn_0_0_outputs)
    num_attn_2_2_output_scores = classifier_weights.loc[
        [("num_attn_2_2_outputs", "_") for v in num_attn_2_2_outputs]
    ].mul(num_attn_2_2_outputs, axis=0)

    # num_attn_2_3 ####################################################
    def num_predicate_2_3(q_num_mlp_0_0_output, k_num_mlp_0_0_output):
        if q_num_mlp_0_0_output in {0, 6, 7}:
            return k_num_mlp_0_0_output == 4
        elif q_num_mlp_0_0_output in {1, 2, 3, 5}:
            return k_num_mlp_0_0_output == 6
        elif q_num_mlp_0_0_output in {4}:
            return k_num_mlp_0_0_output == 3

    num_attn_2_3_pattern = select(
        num_mlp_0_0_outputs, num_mlp_0_0_outputs, num_predicate_2_3
    )
    num_attn_2_3_outputs = aggregate_sum(num_attn_2_3_pattern, ones)
    num_attn_2_3_output_scores = classifier_weights.loc[
        [("num_attn_2_3_outputs", "_") for v in num_attn_2_3_outputs]
    ].mul(num_attn_2_3_outputs, axis=0)

    # mlp_2_0 #####################################################
    def mlp_2_0(num_mlp_0_0_output, attn_2_2_output):
        key = (num_mlp_0_0_output, attn_2_2_output)
        if key in {
            (0, 1),
            (0, 2),
            (0, 5),
            (0, 6),
            (1, 0),
            (1, 6),
            (1, 7),
            (2, 0),
            (2, 6),
            (2, 7),
            (3, 0),
            (3, 6),
            (3, 7),
            (5, 0),
            (5, 6),
            (5, 7),
            (6, 0),
            (6, 2),
            (6, 6),
            (6, 7),
            (7, 0),
            (7, 1),
            (7, 2),
            (7, 5),
            (7, 6),
        }:
            return 5
        elif key in {
            (0, 0),
            (0, 3),
            (0, 7),
            (1, 3),
            (2, 3),
            (3, 3),
            (5, 3),
            (6, 3),
            (7, 3),
            (7, 7),
        }:
            return 7
        return 2

    mlp_2_0_outputs = [
        mlp_2_0(k0, k1) for k0, k1 in zip(num_mlp_0_0_outputs, attn_2_2_outputs)
    ]
    mlp_2_0_output_scores = classifier_weights.loc[
        [("mlp_2_0_outputs", str(v)) for v in mlp_2_0_outputs]
    ]

    # num_mlp_2_0 #################################################
    def num_mlp_2_0(num_attn_1_3_output, num_attn_2_2_output):
        key = (num_attn_1_3_output, num_attn_2_2_output)
        if key in {
            (0, 0),
            (0, 1),
            (0, 2),
            (0, 3),
            (1, 0),
            (1, 1),
            (1, 2),
            (1, 3),
            (2, 0),
            (2, 1),
            (2, 2),
            (2, 3),
            (3, 0),
            (3, 1),
            (3, 2),
            (3, 3),
            (4, 0),
            (4, 1),
            (4, 2),
            (4, 3),
            (5, 0),
            (5, 1),
            (5, 2),
            (5, 3),
            (6, 0),
            (6, 1),
            (6, 2),
            (6, 3),
            (7, 0),
            (7, 1),
            (7, 2),
            (7, 3),
            (8, 0),
            (8, 1),
            (8, 2),
            (8, 3),
            (9, 0),
            (9, 1),
            (9, 2),
            (9, 3),
            (10, 0),
            (10, 1),
            (10, 2),
            (10, 3),
            (11, 0),
            (11, 1),
            (11, 2),
            (11, 3),
            (12, 0),
            (12, 1),
            (12, 2),
            (12, 3),
            (13, 0),
            (13, 1),
            (13, 2),
            (13, 3),
            (14, 0),
            (14, 1),
            (14, 2),
            (14, 3),
            (15, 0),
            (15, 1),
            (15, 2),
            (15, 3),
            (16, 0),
            (16, 1),
            (16, 2),
            (16, 3),
            (17, 0),
            (17, 1),
            (17, 2),
            (17, 3),
            (18, 0),
            (18, 1),
            (18, 2),
            (18, 3),
            (19, 0),
            (19, 1),
            (19, 2),
            (19, 3),
            (19, 4),
            (20, 0),
            (20, 1),
            (20, 2),
            (20, 3),
            (20, 4),
            (21, 0),
            (21, 1),
            (21, 2),
            (21, 3),
            (21, 4),
            (22, 0),
            (22, 1),
            (22, 2),
            (22, 3),
            (22, 4),
            (23, 0),
            (23, 1),
            (23, 2),
            (23, 3),
            (23, 4),
        }:
            return 2
        elif key in {
            (0, 4),
            (0, 5),
            (1, 4),
            (1, 5),
            (2, 4),
            (2, 5),
            (3, 4),
            (3, 5),
            (4, 4),
            (4, 5),
            (5, 4),
            (5, 5),
            (6, 4),
            (6, 5),
            (7, 4),
            (7, 5),
            (7, 6),
            (8, 4),
            (8, 5),
            (8, 6),
            (9, 4),
            (9, 5),
            (9, 6),
            (10, 4),
            (10, 5),
            (10, 6),
            (11, 4),
            (11, 5),
            (11, 6),
            (12, 4),
            (12, 5),
            (12, 6),
            (13, 4),
            (13, 5),
            (13, 6),
            (14, 4),
            (14, 5),
            (14, 6),
            (15, 4),
            (15, 5),
            (15, 6),
            (16, 4),
            (16, 5),
            (16, 6),
            (17, 4),
            (17, 5),
            (17, 6),
            (18, 4),
            (18, 5),
            (18, 6),
            (19, 5),
            (19, 6),
            (20, 5),
            (20, 6),
            (21, 5),
            (21, 6),
            (22, 5),
            (22, 6),
            (23, 5),
            (23, 6),
        }:
            return 4
        return 3

    num_mlp_2_0_outputs = [
        num_mlp_2_0(k0, k1)
        for k0, k1 in zip(num_attn_1_3_outputs, num_attn_2_2_outputs)
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
        ["<s>", "2", "5", "3", "2", "1", "5", "3"],
        ["<pad>", "3", "3", "3", "3", "1", "3", "3"],
    ),
    (["<s>", "2", "4", "3", "5", "2", "5"], ["<pad>", "2", "2", "2", "2", "2", "2"]),
    (["<s>", "5", "2", "0", "4", "3", "4"], ["<pad>", "4", "4", "4", "1", "4", "1"]),
    (
        ["<s>", "5", "3", "4", "1", "2", "5", "2"],
        ["<pad>", "2", "3", "3", "3", "2", "2", "2"],
    ),
    (
        ["<s>", "2", "0", "2", "0", "3", "4", "4"],
        ["<pad>", "3", "3", "3", "3", "1", "3", "3"],
    ),
    (["<s>", "5", "0", "3", "2", "5"], ["<pad>", "1", "3", "3", "3", "1"]),
    (["<s>", "4", "5", "0", "2", "3", "1"], ["<pad>", "6", "6", "6", "6", "6", "6"]),
    (["<s>", "2", "5", "5"], ["<pad>", "1", "1", "1"]),
    (
        ["<s>", "0", "2", "3", "2", "3", "0", "3"],
        ["<pad>", "2", "2", "1", "2", "1", "2", "1"],
    ),
    (
        ["<s>", "2", "1", "1", "2", "3", "3", "4"],
        ["<pad>", "3", "3", "3", "3", "3", "3", "1"],
    ),
]
for x, y in examples:
    print(f"x: {x}")
    print(f"y: {y}")
    y_hat = run(x)
    print(f"y_hat: {y_hat}")
    print()
