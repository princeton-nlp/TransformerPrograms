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


def aggregate(attention, values):
    return [[v for a, v in zip(attn, values) if a][0] for attn in attention]


def run(tokens):
    # classifier weights ##########################################
    classifier_weights = pd.read_csv(
        "programs/rasp_categorical_only/reverse/reverse_weights.csv",
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
        if q_position in {0, 2}:
            return k_position == 7
        elif q_position in {1}:
            return k_position == 6
        elif q_position in {3}:
            return k_position == 4
        elif q_position in {4}:
            return k_position == 3
        elif q_position in {5}:
            return k_position == 2
        elif q_position in {6, 7}:
            return k_position == 1

    attn_0_0_pattern = select_closest(positions, positions, predicate_0_0)
    attn_0_0_outputs = aggregate(attn_0_0_pattern, tokens)
    attn_0_0_output_scores = classifier_weights.loc[
        [("attn_0_0_outputs", str(v)) for v in attn_0_0_outputs]
    ]

    # attn_0_1 ####################################################
    def predicate_0_1(q_position, k_position):
        if q_position in {0}:
            return k_position == 3
        elif q_position in {1}:
            return k_position == 6
        elif q_position in {2, 3}:
            return k_position == 7
        elif q_position in {4, 6, 7}:
            return k_position == 1
        elif q_position in {5}:
            return k_position == 2

    attn_0_1_pattern = select_closest(positions, positions, predicate_0_1)
    attn_0_1_outputs = aggregate(attn_0_1_pattern, tokens)
    attn_0_1_output_scores = classifier_weights.loc[
        [("attn_0_1_outputs", str(v)) for v in attn_0_1_outputs]
    ]

    # attn_0_2 ####################################################
    def predicate_0_2(q_position, k_position):
        if q_position in {0}:
            return k_position == 3
        elif q_position in {1, 2}:
            return k_position == 5
        elif q_position in {3}:
            return k_position == 4
        elif q_position in {4, 5}:
            return k_position == 7
        elif q_position in {6, 7}:
            return k_position == 1

    attn_0_2_pattern = select_closest(positions, positions, predicate_0_2)
    attn_0_2_outputs = aggregate(attn_0_2_pattern, tokens)
    attn_0_2_output_scores = classifier_weights.loc[
        [("attn_0_2_outputs", str(v)) for v in attn_0_2_outputs]
    ]

    # attn_0_3 ####################################################
    def predicate_0_3(q_position, k_position):
        if q_position in {0, 4}:
            return k_position == 3
        elif q_position in {1, 3}:
            return k_position == 4
        elif q_position in {2}:
            return k_position == 5
        elif q_position in {5}:
            return k_position == 2
        elif q_position in {6}:
            return k_position == 1
        elif q_position in {7}:
            return k_position == 7

    attn_0_3_pattern = select_closest(positions, positions, predicate_0_3)
    attn_0_3_outputs = aggregate(attn_0_3_pattern, tokens)
    attn_0_3_output_scores = classifier_weights.loc[
        [("attn_0_3_outputs", str(v)) for v in attn_0_3_outputs]
    ]

    # attn_0_4 ####################################################
    def predicate_0_4(position, token):
        if position in {0}:
            return token == "1"
        elif position in {1, 2, 3}:
            return token == "2"
        elif position in {4, 5}:
            return token == "</s>"
        elif position in {6, 7}:
            return token == "3"

    attn_0_4_pattern = select_closest(tokens, positions, predicate_0_4)
    attn_0_4_outputs = aggregate(attn_0_4_pattern, tokens)
    attn_0_4_output_scores = classifier_weights.loc[
        [("attn_0_4_outputs", str(v)) for v in attn_0_4_outputs]
    ]

    # attn_0_5 ####################################################
    def predicate_0_5(q_position, k_position):
        if q_position in {0, 4}:
            return k_position == 3
        elif q_position in {1, 3}:
            return k_position == 4
        elif q_position in {2}:
            return k_position == 5
        elif q_position in {5}:
            return k_position == 2
        elif q_position in {6, 7}:
            return k_position == 1

    attn_0_5_pattern = select_closest(positions, positions, predicate_0_5)
    attn_0_5_outputs = aggregate(attn_0_5_pattern, tokens)
    attn_0_5_output_scores = classifier_weights.loc[
        [("attn_0_5_outputs", str(v)) for v in attn_0_5_outputs]
    ]

    # attn_0_6 ####################################################
    def predicate_0_6(q_position, k_position):
        if q_position in {0}:
            return k_position == 3
        elif q_position in {1}:
            return k_position == 5
        elif q_position in {2, 3, 4, 5}:
            return k_position == 7
        elif q_position in {6, 7}:
            return k_position == 1

    attn_0_6_pattern = select_closest(positions, positions, predicate_0_6)
    attn_0_6_outputs = aggregate(attn_0_6_pattern, tokens)
    attn_0_6_output_scores = classifier_weights.loc[
        [("attn_0_6_outputs", str(v)) for v in attn_0_6_outputs]
    ]

    # attn_0_7 ####################################################
    def predicate_0_7(q_position, k_position):
        if q_position in {0}:
            return k_position == 5
        elif q_position in {1}:
            return k_position == 6
        elif q_position in {2}:
            return k_position == 2
        elif q_position in {3, 7}:
            return k_position == 4
        elif q_position in {4, 5, 6}:
            return k_position == 1

    attn_0_7_pattern = select_closest(positions, positions, predicate_0_7)
    attn_0_7_outputs = aggregate(attn_0_7_pattern, tokens)
    attn_0_7_output_scores = classifier_weights.loc[
        [("attn_0_7_outputs", str(v)) for v in attn_0_7_outputs]
    ]

    # mlp_0_0 #####################################################
    def mlp_0_0(attn_0_1_output, attn_0_2_output):
        key = (attn_0_1_output, attn_0_2_output)
        if key in {
            ("1", "1"),
            ("4", "0"),
            ("4", "1"),
            ("4", "2"),
            ("4", "4"),
            ("4", "</s>"),
            ("</s>", "4"),
        }:
            return 1
        elif key in {
            ("1", "3"),
            ("1", "4"),
            ("2", "1"),
            ("2", "2"),
            ("2", "3"),
            ("2", "4"),
            ("2", "</s>"),
            ("4", "3"),
        }:
            return 0
        elif key in {
            ("1", "0"),
            ("1", "2"),
            ("1", "</s>"),
            ("2", "0"),
            ("</s>", "0"),
            ("</s>", "1"),
            ("</s>", "2"),
            ("</s>", "</s>"),
        }:
            return 5
        elif key in {("</s>", "3")}:
            return 2
        return 3

    mlp_0_0_outputs = [
        mlp_0_0(k0, k1) for k0, k1 in zip(attn_0_1_outputs, attn_0_2_outputs)
    ]
    mlp_0_0_output_scores = classifier_weights.loc[
        [("mlp_0_0_outputs", str(v)) for v in mlp_0_0_outputs]
    ]

    # mlp_0_1 #####################################################
    def mlp_0_1(attn_0_2_output, attn_0_0_output):
        key = (attn_0_2_output, attn_0_0_output)
        if key in {
            ("1", "1"),
            ("4", "0"),
            ("4", "1"),
            ("4", "2"),
            ("4", "3"),
            ("4", "4"),
            ("4", "<s>"),
            ("<s>", "1"),
        }:
            return 3
        elif key in {
            ("0", "</s>"),
            ("1", "</s>"),
            ("3", "</s>"),
            ("4", "</s>"),
            ("</s>", "0"),
            ("</s>", "1"),
            ("</s>", "3"),
            ("</s>", "4"),
            ("</s>", "</s>"),
            ("<s>", "</s>"),
        }:
            return 0
        elif key in {("2", "</s>"), ("</s>", "2"), ("</s>", "<s>")}:
            return 6
        return 2

    mlp_0_1_outputs = [
        mlp_0_1(k0, k1) for k0, k1 in zip(attn_0_2_outputs, attn_0_0_outputs)
    ]
    mlp_0_1_output_scores = classifier_weights.loc[
        [("mlp_0_1_outputs", str(v)) for v in mlp_0_1_outputs]
    ]

    # mlp_0_2 #####################################################
    def mlp_0_2(attn_0_5_output, attn_0_6_output):
        key = (attn_0_5_output, attn_0_6_output)
        if key in {
            ("0", "2"),
            ("0", "3"),
            ("1", "3"),
            ("2", "3"),
            ("3", "3"),
            ("3", "</s>"),
            ("3", "<s>"),
            ("4", "3"),
            ("</s>", "3"),
            ("<s>", "3"),
        }:
            return 6
        elif key in {
            ("0", "1"),
            ("1", "1"),
            ("1", "</s>"),
            ("1", "<s>"),
            ("2", "1"),
            ("3", "1"),
            ("4", "1"),
            ("</s>", "1"),
            ("<s>", "1"),
        }:
            return 4
        elif key in {
            ("0", "4"),
            ("1", "4"),
            ("2", "4"),
            ("3", "4"),
            ("4", "4"),
            ("4", "</s>"),
            ("4", "<s>"),
            ("</s>", "4"),
            ("<s>", "4"),
        }:
            return 5
        elif key in {
            ("1", "2"),
            ("2", "2"),
            ("2", "</s>"),
            ("2", "<s>"),
            ("3", "2"),
            ("4", "2"),
            ("</s>", "2"),
            ("<s>", "2"),
        }:
            return 2
        return 3

    mlp_0_2_outputs = [
        mlp_0_2(k0, k1) for k0, k1 in zip(attn_0_5_outputs, attn_0_6_outputs)
    ]
    mlp_0_2_output_scores = classifier_weights.loc[
        [("mlp_0_2_outputs", str(v)) for v in mlp_0_2_outputs]
    ]

    # mlp_0_3 #####################################################
    def mlp_0_3(attn_0_5_output, attn_0_2_output):
        key = (attn_0_5_output, attn_0_2_output)
        if key in {
            ("0", "1"),
            ("0", "2"),
            ("1", "2"),
            ("1", "4"),
            ("2", "2"),
            ("2", "4"),
            ("3", "2"),
            ("3", "<s>"),
            ("4", "2"),
            ("4", "3"),
            ("</s>", "2"),
            ("</s>", "4"),
            ("</s>", "</s>"),
            ("</s>", "<s>"),
            ("<s>", "1"),
            ("<s>", "2"),
        }:
            return 0
        elif key in {
            ("0", "3"),
            ("1", "3"),
            ("2", "3"),
            ("3", "3"),
            ("3", "</s>"),
            ("</s>", "3"),
            ("<s>", "3"),
        }:
            return 2
        elif key in {
            ("1", "1"),
            ("1", "</s>"),
            ("1", "<s>"),
            ("2", "1"),
            ("3", "1"),
            ("</s>", "1"),
        }:
            return 3
        elif key in {("2", "<s>")}:
            return 7
        elif key in {("2", "</s>")}:
            return 5
        return 6

    mlp_0_3_outputs = [
        mlp_0_3(k0, k1) for k0, k1 in zip(attn_0_5_outputs, attn_0_2_outputs)
    ]
    mlp_0_3_output_scores = classifier_weights.loc[
        [("mlp_0_3_outputs", str(v)) for v in mlp_0_3_outputs]
    ]

    # attn_1_0 ####################################################
    def predicate_1_0(position, token):
        if position in {0, 4}:
            return token == "3"
        elif position in {1}:
            return token == "</s>"
        elif position in {2}:
            return token == "2"
        elif position in {3}:
            return token == "4"
        elif position in {5, 7}:
            return token == "1"
        elif position in {6}:
            return token == "<s>"

    attn_1_0_pattern = select_closest(tokens, positions, predicate_1_0)
    attn_1_0_outputs = aggregate(attn_1_0_pattern, attn_0_7_outputs)
    attn_1_0_output_scores = classifier_weights.loc[
        [("attn_1_0_outputs", str(v)) for v in attn_1_0_outputs]
    ]

    # attn_1_1 ####################################################
    def predicate_1_1(position, token):
        if position in {0, 3, 7}:
            return token == "1"
        elif position in {1, 4}:
            return token == "</s>"
        elif position in {2}:
            return token == "4"
        elif position in {5}:
            return token == "2"
        elif position in {6}:
            return token == "<s>"

    attn_1_1_pattern = select_closest(tokens, positions, predicate_1_1)
    attn_1_1_outputs = aggregate(attn_1_1_pattern, mlp_0_2_outputs)
    attn_1_1_output_scores = classifier_weights.loc[
        [("attn_1_1_outputs", str(v)) for v in attn_1_1_outputs]
    ]

    # attn_1_2 ####################################################
    def predicate_1_2(q_position, k_position):
        if q_position in {0, 5}:
            return k_position == 5
        elif q_position in {1, 3, 7}:
            return k_position == 3
        elif q_position in {2}:
            return k_position == 2
        elif q_position in {4}:
            return k_position == 6
        elif q_position in {6}:
            return k_position == 0

    attn_1_2_pattern = select_closest(positions, positions, predicate_1_2)
    attn_1_2_outputs = aggregate(attn_1_2_pattern, attn_0_3_outputs)
    attn_1_2_output_scores = classifier_weights.loc[
        [("attn_1_2_outputs", str(v)) for v in attn_1_2_outputs]
    ]

    # attn_1_3 ####################################################
    def predicate_1_3(position, token):
        if position in {0}:
            return token == "0"
        elif position in {1, 2, 4}:
            return token == "3"
        elif position in {3, 5}:
            return token == "2"
        elif position in {6}:
            return token == "<s>"
        elif position in {7}:
            return token == "1"

    attn_1_3_pattern = select_closest(tokens, positions, predicate_1_3)
    attn_1_3_outputs = aggregate(attn_1_3_pattern, attn_0_3_outputs)
    attn_1_3_output_scores = classifier_weights.loc[
        [("attn_1_3_outputs", str(v)) for v in attn_1_3_outputs]
    ]

    # attn_1_4 ####################################################
    def predicate_1_4(attn_0_3_output, token):
        if attn_0_3_output in {"0"}:
            return token == "</s>"
        elif attn_0_3_output in {"1"}:
            return token == "2"
        elif attn_0_3_output in {"2"}:
            return token == "1"
        elif attn_0_3_output in {"<s>", "4", "3"}:
            return token == "4"
        elif attn_0_3_output in {"</s>"}:
            return token == "<s>"

    attn_1_4_pattern = select_closest(tokens, attn_0_3_outputs, predicate_1_4)
    attn_1_4_outputs = aggregate(attn_1_4_pattern, attn_0_3_outputs)
    attn_1_4_output_scores = classifier_weights.loc[
        [("attn_1_4_outputs", str(v)) for v in attn_1_4_outputs]
    ]

    # attn_1_5 ####################################################
    def predicate_1_5(position, token):
        if position in {0}:
            return token == "4"
        elif position in {1, 6}:
            return token == "<s>"
        elif position in {2, 4, 7}:
            return token == "</s>"
        elif position in {3}:
            return token == "3"
        elif position in {5}:
            return token == "2"

    attn_1_5_pattern = select_closest(tokens, positions, predicate_1_5)
    attn_1_5_outputs = aggregate(attn_1_5_pattern, mlp_0_2_outputs)
    attn_1_5_output_scores = classifier_weights.loc[
        [("attn_1_5_outputs", str(v)) for v in attn_1_5_outputs]
    ]

    # attn_1_6 ####################################################
    def predicate_1_6(q_position, k_position):
        if q_position in {0}:
            return k_position == 1
        elif q_position in {1, 6, 7}:
            return k_position == 6
        elif q_position in {2}:
            return k_position == 5
        elif q_position in {3, 4}:
            return k_position == 3
        elif q_position in {5}:
            return k_position == 7

    attn_1_6_pattern = select_closest(positions, positions, predicate_1_6)
    attn_1_6_outputs = aggregate(attn_1_6_pattern, tokens)
    attn_1_6_output_scores = classifier_weights.loc[
        [("attn_1_6_outputs", str(v)) for v in attn_1_6_outputs]
    ]

    # attn_1_7 ####################################################
    def predicate_1_7(position, token):
        if position in {0}:
            return token == "0"
        elif position in {1, 7}:
            return token == "4"
        elif position in {2, 4, 6}:
            return token == "2"
        elif position in {3}:
            return token == "1"
        elif position in {5}:
            return token == "3"

    attn_1_7_pattern = select_closest(tokens, positions, predicate_1_7)
    attn_1_7_outputs = aggregate(attn_1_7_pattern, mlp_0_1_outputs)
    attn_1_7_output_scores = classifier_weights.loc[
        [("attn_1_7_outputs", str(v)) for v in attn_1_7_outputs]
    ]

    # mlp_1_0 #####################################################
    def mlp_1_0(attn_0_0_output, attn_0_2_output):
        key = (attn_0_0_output, attn_0_2_output)
        return 6

    mlp_1_0_outputs = [
        mlp_1_0(k0, k1) for k0, k1 in zip(attn_0_0_outputs, attn_0_2_outputs)
    ]
    mlp_1_0_output_scores = classifier_weights.loc[
        [("mlp_1_0_outputs", str(v)) for v in mlp_1_0_outputs]
    ]

    # mlp_1_1 #####################################################
    def mlp_1_1(attn_0_1_output, attn_0_2_output):
        key = (attn_0_1_output, attn_0_2_output)
        if key in {
            ("0", "1"),
            ("0", "3"),
            ("1", "1"),
            ("1", "2"),
            ("1", "3"),
            ("2", "0"),
            ("2", "1"),
            ("2", "3"),
            ("3", "0"),
            ("3", "1"),
            ("3", "2"),
            ("3", "3"),
            ("3", "4"),
            ("3", "</s>"),
            ("3", "<s>"),
            ("</s>", "1"),
            ("</s>", "3"),
            ("<s>", "1"),
            ("<s>", "3"),
        }:
            return 3
        elif key in {
            ("0", "4"),
            ("4", "0"),
            ("4", "3"),
            ("4", "4"),
            ("</s>", "0"),
            ("</s>", "4"),
        }:
            return 1
        elif key in {("0", "0"), ("1", "0"), ("<s>", "0")}:
            return 7
        elif key in {("1", "4")}:
            return 0
        return 5

    mlp_1_1_outputs = [
        mlp_1_1(k0, k1) for k0, k1 in zip(attn_0_1_outputs, attn_0_2_outputs)
    ]
    mlp_1_1_output_scores = classifier_weights.loc[
        [("mlp_1_1_outputs", str(v)) for v in mlp_1_1_outputs]
    ]

    # mlp_1_2 #####################################################
    def mlp_1_2(attn_0_2_output, attn_0_1_output):
        key = (attn_0_2_output, attn_0_1_output)
        if key in {
            ("0", "0"),
            ("0", "2"),
            ("0", "<s>"),
            ("1", "2"),
            ("1", "<s>"),
            ("2", "0"),
            ("2", "2"),
            ("2", "</s>"),
            ("2", "<s>"),
            ("</s>", "0"),
            ("</s>", "2"),
            ("</s>", "<s>"),
            ("<s>", "0"),
            ("<s>", "2"),
            ("<s>", "<s>"),
        }:
            return 7
        elif key in {
            ("0", "4"),
            ("1", "0"),
            ("1", "1"),
            ("1", "3"),
            ("1", "4"),
            ("1", "</s>"),
            ("3", "4"),
            ("4", "4"),
            ("</s>", "4"),
            ("<s>", "4"),
        }:
            return 1
        elif key in {
            ("0", "</s>"),
            ("3", "</s>"),
            ("4", "</s>"),
            ("</s>", "</s>"),
            ("<s>", "</s>"),
        }:
            return 0
        elif key in {("3", "0"), ("3", "1"), ("3", "2"), ("3", "<s>")}:
            return 6
        return 4

    mlp_1_2_outputs = [
        mlp_1_2(k0, k1) for k0, k1 in zip(attn_0_2_outputs, attn_0_1_outputs)
    ]
    mlp_1_2_output_scores = classifier_weights.loc[
        [("mlp_1_2_outputs", str(v)) for v in mlp_1_2_outputs]
    ]

    # mlp_1_3 #####################################################
    def mlp_1_3(attn_0_3_output, attn_0_6_output):
        key = (attn_0_3_output, attn_0_6_output)
        if key in {("1", "2"), ("1", "3"), ("1", "4"), ("1", "<s>")}:
            return 0
        elif key in {("2", "2"), ("2", "4"), ("2", "<s>")}:
            return 4
        elif key in {("4", "</s>")}:
            return 5
        elif key in {("0", "</s>")}:
            return 6
        return 2

    mlp_1_3_outputs = [
        mlp_1_3(k0, k1) for k0, k1 in zip(attn_0_3_outputs, attn_0_6_outputs)
    ]
    mlp_1_3_output_scores = classifier_weights.loc[
        [("mlp_1_3_outputs", str(v)) for v in mlp_1_3_outputs]
    ]

    # attn_2_0 ####################################################
    def predicate_2_0(q_position, k_position):
        if q_position in {0, 2, 6, 7}:
            return k_position == 1
        elif q_position in {1}:
            return k_position == 2
        elif q_position in {3}:
            return k_position == 3
        elif q_position in {4, 5}:
            return k_position == 7

    attn_2_0_pattern = select_closest(positions, positions, predicate_2_0)
    attn_2_0_outputs = aggregate(attn_2_0_pattern, mlp_0_3_outputs)
    attn_2_0_output_scores = classifier_weights.loc[
        [("attn_2_0_outputs", str(v)) for v in attn_2_0_outputs]
    ]

    # attn_2_1 ####################################################
    def predicate_2_1(position, token):
        if position in {0, 2, 4}:
            return token == "0"
        elif position in {1, 3}:
            return token == "</s>"
        elif position in {5, 6}:
            return token == "<s>"
        elif position in {7}:
            return token == "<pad>"

    attn_2_1_pattern = select_closest(tokens, positions, predicate_2_1)
    attn_2_1_outputs = aggregate(attn_2_1_pattern, attn_1_2_outputs)
    attn_2_1_output_scores = classifier_weights.loc[
        [("attn_2_1_outputs", str(v)) for v in attn_2_1_outputs]
    ]

    # attn_2_2 ####################################################
    def predicate_2_2(mlp_0_1_output, token):
        if mlp_0_1_output in {0}:
            return token == "1"
        elif mlp_0_1_output in {1, 7}:
            return token == "4"
        elif mlp_0_1_output in {2}:
            return token == "<s>"
        elif mlp_0_1_output in {3, 4, 5}:
            return token == "0"
        elif mlp_0_1_output in {6}:
            return token == "3"

    attn_2_2_pattern = select_closest(tokens, mlp_0_1_outputs, predicate_2_2)
    attn_2_2_outputs = aggregate(attn_2_2_pattern, attn_1_6_outputs)
    attn_2_2_output_scores = classifier_weights.loc[
        [("attn_2_2_outputs", str(v)) for v in attn_2_2_outputs]
    ]

    # attn_2_3 ####################################################
    def predicate_2_3(q_position, k_position):
        if q_position in {0, 7}:
            return k_position == 7
        elif q_position in {1, 4}:
            return k_position == 5
        elif q_position in {2}:
            return k_position == 3
        elif q_position in {3}:
            return k_position == 6
        elif q_position in {5, 6}:
            return k_position == 4

    attn_2_3_pattern = select_closest(positions, positions, predicate_2_3)
    attn_2_3_outputs = aggregate(attn_2_3_pattern, attn_1_2_outputs)
    attn_2_3_output_scores = classifier_weights.loc[
        [("attn_2_3_outputs", str(v)) for v in attn_2_3_outputs]
    ]

    # attn_2_4 ####################################################
    def predicate_2_4(position, token):
        if position in {0, 5, 7}:
            return token == "3"
        elif position in {1}:
            return token == "</s>"
        elif position in {2}:
            return token == "4"
        elif position in {3, 4}:
            return token == "<s>"
        elif position in {6}:
            return token == "2"

    attn_2_4_pattern = select_closest(tokens, positions, predicate_2_4)
    attn_2_4_outputs = aggregate(attn_2_4_pattern, attn_1_6_outputs)
    attn_2_4_output_scores = classifier_weights.loc[
        [("attn_2_4_outputs", str(v)) for v in attn_2_4_outputs]
    ]

    # attn_2_5 ####################################################
    def predicate_2_5(q_position, k_position):
        if q_position in {0, 2, 4, 5, 7}:
            return k_position == 7
        elif q_position in {1}:
            return k_position == 4
        elif q_position in {3}:
            return k_position == 3
        elif q_position in {6}:
            return k_position == 1

    attn_2_5_pattern = select_closest(positions, positions, predicate_2_5)
    attn_2_5_outputs = aggregate(attn_2_5_pattern, mlp_0_2_outputs)
    attn_2_5_output_scores = classifier_weights.loc[
        [("attn_2_5_outputs", str(v)) for v in attn_2_5_outputs]
    ]

    # attn_2_6 ####################################################
    def predicate_2_6(q_position, k_position):
        if q_position in {0, 3, 7}:
            return k_position == 3
        elif q_position in {1, 5, 6}:
            return k_position == 5
        elif q_position in {2}:
            return k_position == 1
        elif q_position in {4}:
            return k_position == 6

    attn_2_6_pattern = select_closest(positions, positions, predicate_2_6)
    attn_2_6_outputs = aggregate(attn_2_6_pattern, mlp_0_2_outputs)
    attn_2_6_output_scores = classifier_weights.loc[
        [("attn_2_6_outputs", str(v)) for v in attn_2_6_outputs]
    ]

    # attn_2_7 ####################################################
    def predicate_2_7(mlp_0_2_output, attn_0_5_output):
        if mlp_0_2_output in {0, 1, 3, 5}:
            return attn_0_5_output == "<s>"
        elif mlp_0_2_output in {2, 6, 7}:
            return attn_0_5_output == "</s>"
        elif mlp_0_2_output in {4}:
            return attn_0_5_output == "0"

    attn_2_7_pattern = select_closest(attn_0_5_outputs, mlp_0_2_outputs, predicate_2_7)
    attn_2_7_outputs = aggregate(attn_2_7_pattern, attn_1_6_outputs)
    attn_2_7_output_scores = classifier_weights.loc[
        [("attn_2_7_outputs", str(v)) for v in attn_2_7_outputs]
    ]

    # mlp_2_0 #####################################################
    def mlp_2_0(attn_0_3_output, attn_0_6_output):
        key = (attn_0_3_output, attn_0_6_output)
        if key in {
            ("0", "0"),
            ("0", "2"),
            ("0", "3"),
            ("0", "4"),
            ("0", "<s>"),
            ("2", "0"),
            ("2", "3"),
            ("2", "<s>"),
            ("3", "</s>"),
            ("3", "<s>"),
            ("4", "<s>"),
        }:
            return 5
        elif key in {
            ("2", "</s>"),
            ("4", "</s>"),
            ("</s>", "1"),
            ("</s>", "2"),
            ("</s>", "4"),
            ("</s>", "</s>"),
            ("</s>", "<s>"),
            ("<s>", "1"),
            ("<s>", "</s>"),
        }:
            return 0
        elif key in {
            ("0", "1"),
            ("0", "</s>"),
            ("1", "<s>"),
            ("2", "1"),
            ("<s>", "<s>"),
        }:
            return 6
        return 3

    mlp_2_0_outputs = [
        mlp_2_0(k0, k1) for k0, k1 in zip(attn_0_3_outputs, attn_0_6_outputs)
    ]
    mlp_2_0_output_scores = classifier_weights.loc[
        [("mlp_2_0_outputs", str(v)) for v in mlp_2_0_outputs]
    ]

    # mlp_2_1 #####################################################
    def mlp_2_1(attn_0_2_output, attn_2_3_output):
        key = (attn_0_2_output, attn_2_3_output)
        if key in {
            ("0", "0"),
            ("0", "</s>"),
            ("1", "0"),
            ("1", "</s>"),
            ("2", "0"),
            ("2", "</s>"),
            ("3", "0"),
            ("3", "</s>"),
            ("4", "0"),
            ("4", "</s>"),
            ("</s>", "0"),
            ("</s>", "1"),
            ("</s>", "2"),
            ("</s>", "3"),
            ("</s>", "4"),
            ("</s>", "</s>"),
            ("</s>", "<s>"),
            ("<s>", "0"),
            ("<s>", "</s>"),
        }:
            return 3
        elif key in {
            ("0", "4"),
            ("0", "<s>"),
            ("1", "4"),
            ("2", "4"),
            ("2", "<s>"),
            ("3", "4"),
            ("4", "4"),
            ("<s>", "4"),
        }:
            return 7
        return 4

    mlp_2_1_outputs = [
        mlp_2_1(k0, k1) for k0, k1 in zip(attn_0_2_outputs, attn_2_3_outputs)
    ]
    mlp_2_1_output_scores = classifier_weights.loc[
        [("mlp_2_1_outputs", str(v)) for v in mlp_2_1_outputs]
    ]

    # mlp_2_2 #####################################################
    def mlp_2_2(attn_0_2_output, attn_0_0_output):
        key = (attn_0_2_output, attn_0_0_output)
        if key in {
            ("0", "1"),
            ("1", "1"),
            ("1", "</s>"),
            ("1", "<s>"),
            ("2", "1"),
            ("2", "</s>"),
            ("3", "1"),
            ("4", "1"),
            ("4", "</s>"),
            ("</s>", "1"),
            ("</s>", "</s>"),
        }:
            return 4
        elif key in {
            ("0", "0"),
            ("0", "2"),
            ("0", "4"),
            ("1", "4"),
            ("2", "4"),
            ("3", "0"),
            ("3", "2"),
            ("3", "4"),
            ("</s>", "4"),
            ("<s>", "0"),
        }:
            return 7
        elif key in {
            ("0", "<s>"),
            ("3", "<s>"),
            ("<s>", "1"),
            ("<s>", "2"),
            ("<s>", "<s>"),
        }:
            return 0
        elif key in {
            ("1", "2"),
            ("2", "2"),
            ("2", "<s>"),
            ("</s>", "2"),
            ("</s>", "<s>"),
        }:
            return 3
        elif key in {("1", "0"), ("2", "0"), ("</s>", "0")}:
            return 5
        return 2

    mlp_2_2_outputs = [
        mlp_2_2(k0, k1) for k0, k1 in zip(attn_0_2_outputs, attn_0_0_outputs)
    ]
    mlp_2_2_output_scores = classifier_weights.loc[
        [("mlp_2_2_outputs", str(v)) for v in mlp_2_2_outputs]
    ]

    # mlp_2_3 #####################################################
    def mlp_2_3(attn_0_3_output, attn_0_2_output):
        key = (attn_0_3_output, attn_0_2_output)
        if key in {
            ("0", "0"),
            ("0", "</s>"),
            ("0", "<s>"),
            ("1", "</s>"),
            ("</s>", "0"),
            ("<s>", "0"),
        }:
            return 2
        elif key in {("2", "0"), ("3", "0"), ("4", "0")}:
            return 7
        elif key in {("4", "</s>")}:
            return 0
        return 6

    mlp_2_3_outputs = [
        mlp_2_3(k0, k1) for k0, k1 in zip(attn_0_3_outputs, attn_0_2_outputs)
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
        ["<s>", "3", "4", "2", "3", "1", "</s>"],
        ["<pad>", "1", "3", "2", "4", "3", "<pad>"],
    ),
    (
        ["<s>", "1", "3", "3", "2", "0", "2", "</s>"],
        ["<pad>", "2", "0", "2", "3", "3", "1", "<pad>"],
    ),
    (
        ["<s>", "4", "4", "4", "1", "1", "4", "</s>"],
        ["<pad>", "4", "1", "1", "4", "4", "4", "<pad>"],
    ),
    (
        ["<s>", "1", "0", "2", "3", "1", "2", "</s>"],
        ["<pad>", "2", "1", "3", "2", "0", "1", "<pad>"],
    ),
    (
        ["<s>", "3", "0", "0", "0", "3", "</s>"],
        ["<pad>", "3", "0", "0", "0", "3", "<pad>"],
    ),
    (
        ["<s>", "4", "2", "1", "1", "2", "3", "</s>"],
        ["<pad>", "3", "2", "1", "1", "2", "4", "<pad>"],
    ),
    (
        ["<s>", "1", "0", "1", "2", "1", "3", "</s>"],
        ["<pad>", "3", "1", "2", "1", "0", "1", "<pad>"],
    ),
    (
        ["<s>", "1", "4", "2", "4", "1", "</s>"],
        ["<pad>", "1", "4", "2", "4", "1", "<pad>"],
    ),
    (
        ["<s>", "2", "3", "1", "2", "4", "1", "</s>"],
        ["<pad>", "1", "4", "2", "1", "3", "2", "<pad>"],
    ),
    (
        ["<s>", "2", "3", "3", "0", "1", "0", "</s>"],
        ["<pad>", "0", "1", "0", "3", "3", "2", "<pad>"],
    ),
]
for x, y in examples:
    print(f"x: {x}")
    print(f"y: {y}")
    y_hat = run(x)
    print(f"y_hat: {y_hat}")
    print()
