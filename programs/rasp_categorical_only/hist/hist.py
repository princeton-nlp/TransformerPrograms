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
        "programs/rasp_categorical_only/hist/hist_weights.csv",
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
    def predicate_0_0(q_token, k_token):
        if q_token in {"0", "2"}:
            return k_token == "4"
        elif q_token in {"1", "3"}:
            return k_token == "1"
        elif q_token in {"4"}:
            return k_token == "3"
        elif q_token in {"5"}:
            return k_token == "5"
        elif q_token in {"<s>"}:
            return k_token == "2"

    attn_0_0_pattern = select_closest(tokens, tokens, predicate_0_0)
    attn_0_0_outputs = aggregate(attn_0_0_pattern, tokens)
    attn_0_0_output_scores = classifier_weights.loc[
        [("attn_0_0_outputs", str(v)) for v in attn_0_0_outputs]
    ]

    # attn_0_1 ####################################################
    def predicate_0_1(q_position, k_position):
        if q_position in {0}:
            return k_position == 0
        elif q_position in {1, 5}:
            return k_position == 4
        elif q_position in {2, 6}:
            return k_position == 3
        elif q_position in {3, 7}:
            return k_position == 2
        elif q_position in {4}:
            return k_position == 1

    attn_0_1_pattern = select_closest(positions, positions, predicate_0_1)
    attn_0_1_outputs = aggregate(attn_0_1_pattern, tokens)
    attn_0_1_output_scores = classifier_weights.loc[
        [("attn_0_1_outputs", str(v)) for v in attn_0_1_outputs]
    ]

    # attn_0_2 ####################################################
    def predicate_0_2(q_position, k_position):
        if q_position in {0}:
            return k_position == 5
        elif q_position in {1, 7}:
            return k_position == 3
        elif q_position in {2, 4, 5}:
            return k_position == 6
        elif q_position in {3, 6}:
            return k_position == 1

    attn_0_2_pattern = select_closest(positions, positions, predicate_0_2)
    attn_0_2_outputs = aggregate(attn_0_2_pattern, tokens)
    attn_0_2_output_scores = classifier_weights.loc[
        [("attn_0_2_outputs", str(v)) for v in attn_0_2_outputs]
    ]

    # attn_0_3 ####################################################
    def predicate_0_3(q_token, k_token):
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

    attn_0_3_pattern = select_closest(tokens, tokens, predicate_0_3)
    attn_0_3_outputs = aggregate(attn_0_3_pattern, positions)
    attn_0_3_output_scores = classifier_weights.loc[
        [("attn_0_3_outputs", str(v)) for v in attn_0_3_outputs]
    ]

    # attn_0_4 ####################################################
    def predicate_0_4(q_token, k_token):
        if q_token in {"5", "0", "2"}:
            return k_token == "3"
        elif q_token in {"1", "3"}:
            return k_token == "4"
        elif q_token in {"4"}:
            return k_token == "2"
        elif q_token in {"<s>"}:
            return k_token == "<pad>"

    attn_0_4_pattern = select_closest(tokens, tokens, predicate_0_4)
    attn_0_4_outputs = aggregate(attn_0_4_pattern, positions)
    attn_0_4_output_scores = classifier_weights.loc[
        [("attn_0_4_outputs", str(v)) for v in attn_0_4_outputs]
    ]

    # attn_0_5 ####################################################
    def predicate_0_5(q_token, k_token):
        if q_token in {"<s>", "0"}:
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

    attn_0_5_pattern = select_closest(tokens, tokens, predicate_0_5)
    attn_0_5_outputs = aggregate(attn_0_5_pattern, tokens)
    attn_0_5_output_scores = classifier_weights.loc[
        [("attn_0_5_outputs", str(v)) for v in attn_0_5_outputs]
    ]

    # attn_0_6 ####################################################
    def predicate_0_6(q_token, k_token):
        if q_token in {"0"}:
            return k_token == "0"
        elif q_token in {"1", "<s>"}:
            return k_token == "1"
        elif q_token in {"2"}:
            return k_token == "2"
        elif q_token in {"3"}:
            return k_token == "3"
        elif q_token in {"4"}:
            return k_token == "4"
        elif q_token in {"5"}:
            return k_token == "5"

    attn_0_6_pattern = select_closest(tokens, tokens, predicate_0_6)
    attn_0_6_outputs = aggregate(attn_0_6_pattern, tokens)
    attn_0_6_output_scores = classifier_weights.loc[
        [("attn_0_6_outputs", str(v)) for v in attn_0_6_outputs]
    ]

    # attn_0_7 ####################################################
    def predicate_0_7(q_token, k_token):
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

    attn_0_7_pattern = select_closest(tokens, tokens, predicate_0_7)
    attn_0_7_outputs = aggregate(attn_0_7_pattern, positions)
    attn_0_7_output_scores = classifier_weights.loc[
        [("attn_0_7_outputs", str(v)) for v in attn_0_7_outputs]
    ]

    # mlp_0_0 #####################################################
    def mlp_0_0(attn_0_6_output, attn_0_2_output):
        key = (attn_0_6_output, attn_0_2_output)
        if key in {
            ("<s>", "0"),
            ("<s>", "1"),
            ("<s>", "2"),
            ("<s>", "3"),
            ("<s>", "4"),
            ("<s>", "5"),
            ("<s>", "<s>"),
        }:
            return 3
        elif key in {("1", "1"), ("2", "2"), ("5", "5")}:
            return 6
        return 4

    mlp_0_0_outputs = [
        mlp_0_0(k0, k1) for k0, k1 in zip(attn_0_6_outputs, attn_0_2_outputs)
    ]
    mlp_0_0_output_scores = classifier_weights.loc[
        [("mlp_0_0_outputs", str(v)) for v in mlp_0_0_outputs]
    ]

    # mlp_0_1 #####################################################
    def mlp_0_1(attn_0_5_output, attn_0_2_output):
        key = (attn_0_5_output, attn_0_2_output)
        if key in {("0", "0"), ("3", "3"), ("4", "4"), ("5", "5")}:
            return 1
        elif key in {("3", "5"), ("5", "3")}:
            return 3
        return 0

    mlp_0_1_outputs = [
        mlp_0_1(k0, k1) for k0, k1 in zip(attn_0_5_outputs, attn_0_2_outputs)
    ]
    mlp_0_1_output_scores = classifier_weights.loc[
        [("mlp_0_1_outputs", str(v)) for v in mlp_0_1_outputs]
    ]

    # mlp_0_2 #####################################################
    def mlp_0_2(attn_0_5_output, attn_0_1_output):
        key = (attn_0_5_output, attn_0_1_output)
        if key in {
            ("0", "0"),
            ("0", "1"),
            ("0", "2"),
            ("0", "3"),
            ("0", "4"),
            ("0", "5"),
            ("0", "<s>"),
            ("2", "5"),
            ("3", "0"),
            ("3", "5"),
            ("4", "0"),
        }:
            return 0
        return 7

    mlp_0_2_outputs = [
        mlp_0_2(k0, k1) for k0, k1 in zip(attn_0_5_outputs, attn_0_1_outputs)
    ]
    mlp_0_2_output_scores = classifier_weights.loc[
        [("mlp_0_2_outputs", str(v)) for v in mlp_0_2_outputs]
    ]

    # mlp_0_3 #####################################################
    def mlp_0_3(attn_0_1_output, attn_0_0_output):
        key = (attn_0_1_output, attn_0_0_output)
        if key in {
            ("0", "0"),
            ("0", "1"),
            ("0", "4"),
            ("0", "<s>"),
            ("1", "1"),
            ("1", "<s>"),
            ("2", "<s>"),
            ("3", "1"),
            ("3", "<s>"),
            ("4", "0"),
            ("4", "1"),
            ("4", "4"),
            ("4", "<s>"),
            ("5", "1"),
            ("5", "<s>"),
            ("<s>", "0"),
            ("<s>", "1"),
            ("<s>", "4"),
            ("<s>", "<s>"),
        }:
            return 2
        elif key in {("4", "3"), ("4", "5")}:
            return 4
        return 1

    mlp_0_3_outputs = [
        mlp_0_3(k0, k1) for k0, k1 in zip(attn_0_1_outputs, attn_0_0_outputs)
    ]
    mlp_0_3_output_scores = classifier_weights.loc[
        [("mlp_0_3_outputs", str(v)) for v in mlp_0_3_outputs]
    ]

    # attn_1_0 ####################################################
    def predicate_1_0(attn_0_4_output, mlp_0_3_output):
        if attn_0_4_output in {0, 1, 2, 3}:
            return mlp_0_3_output == 2
        elif attn_0_4_output in {4, 7}:
            return mlp_0_3_output == 7
        elif attn_0_4_output in {5, 6}:
            return mlp_0_3_output == 1

    attn_1_0_pattern = select_closest(mlp_0_3_outputs, attn_0_4_outputs, predicate_1_0)
    attn_1_0_outputs = aggregate(attn_1_0_pattern, attn_0_4_outputs)
    attn_1_0_output_scores = classifier_weights.loc[
        [("attn_1_0_outputs", str(v)) for v in attn_1_0_outputs]
    ]

    # attn_1_1 ####################################################
    def predicate_1_1(q_token, k_token):
        if q_token in {"1", "4", "<s>", "0", "2", "3", "5"}:
            return k_token == "<s>"

    attn_1_1_pattern = select_closest(tokens, tokens, predicate_1_1)
    attn_1_1_outputs = aggregate(attn_1_1_pattern, mlp_0_2_outputs)
    attn_1_1_output_scores = classifier_weights.loc[
        [("attn_1_1_outputs", str(v)) for v in attn_1_1_outputs]
    ]

    # attn_1_2 ####################################################
    def predicate_1_2(q_mlp_0_1_output, k_mlp_0_1_output):
        if q_mlp_0_1_output in {0}:
            return k_mlp_0_1_output == 3
        elif q_mlp_0_1_output in {1, 2, 3, 4, 5, 7}:
            return k_mlp_0_1_output == 0
        elif q_mlp_0_1_output in {6}:
            return k_mlp_0_1_output == 6

    attn_1_2_pattern = select_closest(mlp_0_1_outputs, mlp_0_1_outputs, predicate_1_2)
    attn_1_2_outputs = aggregate(attn_1_2_pattern, attn_0_4_outputs)
    attn_1_2_output_scores = classifier_weights.loc[
        [("attn_1_2_outputs", str(v)) for v in attn_1_2_outputs]
    ]

    # attn_1_3 ####################################################
    def predicate_1_3(attn_0_1_output, mlp_0_0_output):
        if attn_0_1_output in {"1", "<s>", "0", "2", "3", "5"}:
            return mlp_0_0_output == 7
        elif attn_0_1_output in {"4"}:
            return mlp_0_0_output == 0

    attn_1_3_pattern = select_closest(mlp_0_0_outputs, attn_0_1_outputs, predicate_1_3)
    attn_1_3_outputs = aggregate(attn_1_3_pattern, mlp_0_3_outputs)
    attn_1_3_output_scores = classifier_weights.loc[
        [("attn_1_3_outputs", str(v)) for v in attn_1_3_outputs]
    ]

    # attn_1_4 ####################################################
    def predicate_1_4(attn_0_7_output, position):
        if attn_0_7_output in {0, 1, 2, 3, 6, 7}:
            return position == 6
        elif attn_0_7_output in {4, 5}:
            return position == 4

    attn_1_4_pattern = select_closest(positions, attn_0_7_outputs, predicate_1_4)
    attn_1_4_outputs = aggregate(attn_1_4_pattern, positions)
    attn_1_4_output_scores = classifier_weights.loc[
        [("attn_1_4_outputs", str(v)) for v in attn_1_4_outputs]
    ]

    # attn_1_5 ####################################################
    def predicate_1_5(q_token, k_token):
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

    attn_1_5_pattern = select_closest(tokens, tokens, predicate_1_5)
    attn_1_5_outputs = aggregate(attn_1_5_pattern, mlp_0_1_outputs)
    attn_1_5_output_scores = classifier_weights.loc[
        [("attn_1_5_outputs", str(v)) for v in attn_1_5_outputs]
    ]

    # attn_1_6 ####################################################
    def predicate_1_6(q_token, k_token):
        if q_token in {"0", "2"}:
            return k_token == "4"
        elif q_token in {"1", "<s>", "5", "4"}:
            return k_token == "5"
        elif q_token in {"3"}:
            return k_token == "3"

    attn_1_6_pattern = select_closest(tokens, tokens, predicate_1_6)
    attn_1_6_outputs = aggregate(attn_1_6_pattern, positions)
    attn_1_6_output_scores = classifier_weights.loc[
        [("attn_1_6_outputs", str(v)) for v in attn_1_6_outputs]
    ]

    # attn_1_7 ####################################################
    def predicate_1_7(attn_0_5_output, token):
        if attn_0_5_output in {"0"}:
            return token == "0"
        elif attn_0_5_output in {"1"}:
            return token == "1"
        elif attn_0_5_output in {"2"}:
            return token == "2"
        elif attn_0_5_output in {"3"}:
            return token == "3"
        elif attn_0_5_output in {"4"}:
            return token == "4"
        elif attn_0_5_output in {"5"}:
            return token == "5"
        elif attn_0_5_output in {"<s>"}:
            return token == "<s>"

    attn_1_7_pattern = select_closest(tokens, attn_0_5_outputs, predicate_1_7)
    attn_1_7_outputs = aggregate(attn_1_7_pattern, positions)
    attn_1_7_output_scores = classifier_weights.loc[
        [("attn_1_7_outputs", str(v)) for v in attn_1_7_outputs]
    ]

    # mlp_1_0 #####################################################
    def mlp_1_0(attn_0_2_output, attn_0_0_output):
        key = (attn_0_2_output, attn_0_0_output)
        if key in {
            ("1", "1"),
            ("2", "2"),
            ("3", "3"),
            ("3", "<s>"),
            ("4", "3"),
            ("4", "4"),
            ("4", "<s>"),
            ("<s>", "0"),
            ("<s>", "1"),
            ("<s>", "3"),
            ("<s>", "4"),
            ("<s>", "5"),
            ("<s>", "<s>"),
        }:
            return 6
        return 7

    mlp_1_0_outputs = [
        mlp_1_0(k0, k1) for k0, k1 in zip(attn_0_2_outputs, attn_0_0_outputs)
    ]
    mlp_1_0_output_scores = classifier_weights.loc[
        [("mlp_1_0_outputs", str(v)) for v in mlp_1_0_outputs]
    ]

    # mlp_1_1 #####################################################
    def mlp_1_1(attn_0_6_output, attn_0_1_output):
        key = (attn_0_6_output, attn_0_1_output)
        if key in {("1", "1"), ("2", "2"), ("4", "4"), ("5", "5"), ("<s>", "5")}:
            return 3
        elif key in {("3", "3")}:
            return 6
        elif key in {("0", "0"), ("<s>", "0")}:
            return 7
        return 4

    mlp_1_1_outputs = [
        mlp_1_1(k0, k1) for k0, k1 in zip(attn_0_6_outputs, attn_0_1_outputs)
    ]
    mlp_1_1_output_scores = classifier_weights.loc[
        [("mlp_1_1_outputs", str(v)) for v in mlp_1_1_outputs]
    ]

    # mlp_1_2 #####################################################
    def mlp_1_2(attn_0_5_output, attn_0_6_output):
        key = (attn_0_5_output, attn_0_6_output)
        if key in {
            ("0", "0"),
            ("0", "2"),
            ("0", "4"),
            ("2", "0"),
            ("2", "1"),
            ("2", "3"),
            ("2", "4"),
            ("2", "<s>"),
            ("4", "0"),
            ("4", "2"),
            ("4", "4"),
            ("5", "0"),
            ("5", "4"),
            ("<s>", "0"),
            ("<s>", "4"),
        }:
            return 7
        return 5

    mlp_1_2_outputs = [
        mlp_1_2(k0, k1) for k0, k1 in zip(attn_0_5_outputs, attn_0_6_outputs)
    ]
    mlp_1_2_output_scores = classifier_weights.loc[
        [("mlp_1_2_outputs", str(v)) for v in mlp_1_2_outputs]
    ]

    # mlp_1_3 #####################################################
    def mlp_1_3(attn_0_7_output, position):
        key = (attn_0_7_output, position)
        if key in {
            (2, 6),
            (2, 7),
            (4, 2),
            (4, 6),
            (5, 1),
            (5, 2),
            (5, 3),
            (5, 4),
            (5, 7),
            (6, 1),
            (6, 2),
            (6, 3),
            (6, 4),
            (7, 2),
        }:
            return 7
        elif key in {
            (1, 3),
            (1, 6),
            (1, 7),
            (3, 1),
            (3, 2),
            (3, 6),
            (3, 7),
            (4, 1),
            (4, 7),
            (7, 3),
        }:
            return 5
        elif key in {(1, 2), (1, 5), (2, 1), (2, 4), (2, 5)}:
            return 1
        elif key in {(3, 4), (3, 5), (4, 3)}:
            return 6
        return 4

    mlp_1_3_outputs = [mlp_1_3(k0, k1) for k0, k1 in zip(attn_0_7_outputs, positions)]
    mlp_1_3_output_scores = classifier_weights.loc[
        [("mlp_1_3_outputs", str(v)) for v in mlp_1_3_outputs]
    ]

    # attn_2_0 ####################################################
    def predicate_2_0(attn_0_1_output, mlp_0_0_output):
        if attn_0_1_output in {"<s>", "0", "4"}:
            return mlp_0_0_output == 7
        elif attn_0_1_output in {"1"}:
            return mlp_0_0_output == 2
        elif attn_0_1_output in {"2"}:
            return mlp_0_0_output == 3
        elif attn_0_1_output in {"3"}:
            return mlp_0_0_output == 1
        elif attn_0_1_output in {"5"}:
            return mlp_0_0_output == 5

    attn_2_0_pattern = select_closest(mlp_0_0_outputs, attn_0_1_outputs, predicate_2_0)
    attn_2_0_outputs = aggregate(attn_2_0_pattern, mlp_0_0_outputs)
    attn_2_0_output_scores = classifier_weights.loc[
        [("attn_2_0_outputs", str(v)) for v in attn_2_0_outputs]
    ]

    # attn_2_1 ####################################################
    def predicate_2_1(attn_0_3_output, position):
        if attn_0_3_output in {0, 1, 3, 4, 6}:
            return position == 7
        elif attn_0_3_output in {2, 5, 7}:
            return position == 6

    attn_2_1_pattern = select_closest(positions, attn_0_3_outputs, predicate_2_1)
    attn_2_1_outputs = aggregate(attn_2_1_pattern, positions)
    attn_2_1_output_scores = classifier_weights.loc[
        [("attn_2_1_outputs", str(v)) for v in attn_2_1_outputs]
    ]

    # attn_2_2 ####################################################
    def predicate_2_2(mlp_0_2_output, token):
        if mlp_0_2_output in {0, 1, 2, 3, 5, 6, 7}:
            return token == "<s>"
        elif mlp_0_2_output in {4}:
            return token == "0"

    attn_2_2_pattern = select_closest(tokens, mlp_0_2_outputs, predicate_2_2)
    attn_2_2_outputs = aggregate(attn_2_2_pattern, attn_1_5_outputs)
    attn_2_2_output_scores = classifier_weights.loc[
        [("attn_2_2_outputs", str(v)) for v in attn_2_2_outputs]
    ]

    # attn_2_3 ####################################################
    def predicate_2_3(attn_0_3_output, mlp_0_2_output):
        if attn_0_3_output in {0, 3}:
            return mlp_0_2_output == 2
        elif attn_0_3_output in {1, 2, 6, 7}:
            return mlp_0_2_output == 6
        elif attn_0_3_output in {4, 5}:
            return mlp_0_2_output == 7

    attn_2_3_pattern = select_closest(mlp_0_2_outputs, attn_0_3_outputs, predicate_2_3)
    attn_2_3_outputs = aggregate(attn_2_3_pattern, positions)
    attn_2_3_output_scores = classifier_weights.loc[
        [("attn_2_3_outputs", str(v)) for v in attn_2_3_outputs]
    ]

    # attn_2_4 ####################################################
    def predicate_2_4(mlp_0_3_output, mlp_0_2_output):
        if mlp_0_3_output in {0, 2, 3, 7}:
            return mlp_0_2_output == 6
        elif mlp_0_3_output in {1}:
            return mlp_0_2_output == 4
        elif mlp_0_3_output in {4, 5, 6}:
            return mlp_0_2_output == 7

    attn_2_4_pattern = select_closest(mlp_0_2_outputs, mlp_0_3_outputs, predicate_2_4)
    attn_2_4_outputs = aggregate(attn_2_4_pattern, attn_1_1_outputs)
    attn_2_4_output_scores = classifier_weights.loc[
        [("attn_2_4_outputs", str(v)) for v in attn_2_4_outputs]
    ]

    # attn_2_5 ####################################################
    def predicate_2_5(q_position, k_position):
        if q_position in {0, 1, 3, 5, 6, 7}:
            return k_position == 7
        elif q_position in {2, 4}:
            return k_position == 5

    attn_2_5_pattern = select_closest(positions, positions, predicate_2_5)
    attn_2_5_outputs = aggregate(attn_2_5_pattern, positions)
    attn_2_5_output_scores = classifier_weights.loc[
        [("attn_2_5_outputs", str(v)) for v in attn_2_5_outputs]
    ]

    # attn_2_6 ####################################################
    def predicate_2_6(q_token, k_token):
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

    attn_2_6_pattern = select_closest(tokens, tokens, predicate_2_6)
    attn_2_6_outputs = aggregate(attn_2_6_pattern, mlp_0_0_outputs)
    attn_2_6_output_scores = classifier_weights.loc[
        [("attn_2_6_outputs", str(v)) for v in attn_2_6_outputs]
    ]

    # attn_2_7 ####################################################
    def predicate_2_7(attn_1_3_output, mlp_0_0_output):
        if attn_1_3_output in {0}:
            return mlp_0_0_output == 6
        elif attn_1_3_output in {1, 3, 4, 5, 6, 7}:
            return mlp_0_0_output == 7
        elif attn_1_3_output in {2}:
            return mlp_0_0_output == 1

    attn_2_7_pattern = select_closest(mlp_0_0_outputs, attn_1_3_outputs, predicate_2_7)
    attn_2_7_outputs = aggregate(attn_2_7_pattern, attn_1_6_outputs)
    attn_2_7_output_scores = classifier_weights.loc[
        [("attn_2_7_outputs", str(v)) for v in attn_2_7_outputs]
    ]

    # mlp_2_0 #####################################################
    def mlp_2_0(attn_2_5_output, attn_2_1_output):
        key = (attn_2_5_output, attn_2_1_output)
        if key in {
            (0, 0),
            (0, 1),
            (0, 2),
            (0, 3),
            (0, 4),
            (0, 7),
            (1, 0),
            (1, 1),
            (1, 3),
            (1, 4),
            (1, 7),
            (2, 0),
            (2, 1),
            (2, 2),
            (2, 3),
            (2, 4),
            (2, 7),
            (3, 0),
            (3, 1),
            (3, 3),
            (3, 4),
            (3, 7),
            (4, 0),
            (4, 1),
            (4, 3),
            (4, 4),
            (4, 7),
        }:
            return 1
        elif key in {(0, 5)}:
            return 7
        return 3

    mlp_2_0_outputs = [
        mlp_2_0(k0, k1) for k0, k1 in zip(attn_2_5_outputs, attn_2_1_outputs)
    ]
    mlp_2_0_output_scores = classifier_weights.loc[
        [("mlp_2_0_outputs", str(v)) for v in mlp_2_0_outputs]
    ]

    # mlp_2_1 #####################################################
    def mlp_2_1(attn_1_5_output):
        key = attn_1_5_output
        return 1

    mlp_2_1_outputs = [mlp_2_1(k0) for k0 in attn_1_5_outputs]
    mlp_2_1_output_scores = classifier_weights.loc[
        [("mlp_2_1_outputs", str(v)) for v in mlp_2_1_outputs]
    ]

    # mlp_2_2 #####################################################
    def mlp_2_2(attn_2_6_output, attn_1_1_output):
        key = (attn_2_6_output, attn_1_1_output)
        if key in {
            (0, 2),
            (0, 4),
            (0, 7),
            (1, 7),
            (2, 2),
            (2, 7),
            (3, 7),
            (4, 0),
            (4, 1),
            (4, 2),
            (4, 3),
            (4, 4),
            (4, 5),
            (4, 6),
            (4, 7),
            (5, 7),
            (6, 7),
            (7, 0),
            (7, 2),
            (7, 5),
            (7, 7),
        }:
            return 0
        elif key in {
            (3, 0),
            (3, 1),
            (3, 2),
            (3, 3),
            (3, 4),
            (3, 5),
            (7, 1),
            (7, 3),
            (7, 4),
            (7, 6),
        }:
            return 4
        elif key in {(0, 0), (0, 1), (0, 3), (2, 4), (5, 2)}:
            return 5
        return 1

    mlp_2_2_outputs = [
        mlp_2_2(k0, k1) for k0, k1 in zip(attn_2_6_outputs, attn_1_1_outputs)
    ]
    mlp_2_2_output_scores = classifier_weights.loc[
        [("mlp_2_2_outputs", str(v)) for v in mlp_2_2_outputs]
    ]

    # mlp_2_3 #####################################################
    def mlp_2_3(attn_0_6_output, mlp_0_0_output):
        key = (attn_0_6_output, mlp_0_0_output)
        return 1

    mlp_2_3_outputs = [
        mlp_2_3(k0, k1) for k0, k1 in zip(attn_0_6_outputs, mlp_0_0_outputs)
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
    (["<s>", "1", "5", "1", "2", "0", "3"], ["<pad>", "2", "1", "2", "1", "1", "1"]),
    (
        ["<s>", "0", "1", "3", "1", "3", "3", "5"],
        ["<pad>", "1", "2", "3", "2", "3", "3", "1"],
    ),
    (
        ["<s>", "5", "5", "3", "0", "5", "1", "4"],
        ["<pad>", "3", "3", "1", "1", "3", "1", "1"],
    ),
    (
        ["<s>", "1", "2", "5", "5", "0", "2", "3"],
        ["<pad>", "1", "2", "2", "2", "1", "2", "1"],
    ),
    (
        ["<s>", "2", "3", "4", "3", "2", "2", "2"],
        ["<pad>", "4", "2", "1", "2", "4", "4", "4"],
    ),
    (
        ["<s>", "3", "5", "0", "4", "4", "5", "1"],
        ["<pad>", "1", "2", "1", "2", "2", "2", "1"],
    ),
    (["<s>", "2", "0", "2", "4", "5", "0"], ["<pad>", "2", "2", "2", "1", "1", "2"]),
    (["<s>", "1", "2", "4", "3"], ["<pad>", "1", "1", "1", "1"]),
    (["<s>", "0", "2", "1", "4", "0"], ["<pad>", "2", "1", "1", "1", "2"]),
    (["<s>", "5", "4", "2", "5", "2", "1"], ["<pad>", "2", "1", "2", "2", "2", "1"]),
]
for x, y in examples:
    print(f"x: {x}")
    print(f"y: {y}")
    y_hat = run(x)
    print(f"y_hat: {y_hat}")
    print()
