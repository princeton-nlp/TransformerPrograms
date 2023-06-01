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
        "programs/rasp_categorical_only/sort/sort_weights.csv",
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
        if q_token in {"<s>", "0"}:
            return k_token == "0"
        elif q_token in {"1"}:
            return k_token == "1"
        elif q_token in {"2"}:
            return k_token == "2"
        elif q_token in {"3"}:
            return k_token == "3"
        elif q_token in {"4", "</s>"}:
            return k_token == "4"

    attn_0_0_pattern = select_closest(tokens, tokens, predicate_0_0)
    attn_0_0_outputs = aggregate(attn_0_0_pattern, tokens)
    attn_0_0_output_scores = classifier_weights.loc[
        [("attn_0_0_outputs", str(v)) for v in attn_0_0_outputs]
    ]

    # attn_0_1 ####################################################
    def predicate_0_1(q_position, k_position):
        if q_position in {0}:
            return k_position == 1
        elif q_position in {1, 6}:
            return k_position == 2
        elif q_position in {2}:
            return k_position == 3
        elif q_position in {3, 5}:
            return k_position == 4
        elif q_position in {4, 7}:
            return k_position == 6

    attn_0_1_pattern = select_closest(positions, positions, predicate_0_1)
    attn_0_1_outputs = aggregate(attn_0_1_pattern, tokens)
    attn_0_1_output_scores = classifier_weights.loc[
        [("attn_0_1_outputs", str(v)) for v in attn_0_1_outputs]
    ]

    # attn_0_2 ####################################################
    def predicate_0_2(q_position, k_position):
        if q_position in {0}:
            return k_position == 1
        elif q_position in {1}:
            return k_position == 3
        elif q_position in {2, 6}:
            return k_position == 4
        elif q_position in {3, 4}:
            return k_position == 5
        elif q_position in {5, 7}:
            return k_position == 6

    attn_0_2_pattern = select_closest(positions, positions, predicate_0_2)
    attn_0_2_outputs = aggregate(attn_0_2_pattern, tokens)
    attn_0_2_output_scores = classifier_weights.loc[
        [("attn_0_2_outputs", str(v)) for v in attn_0_2_outputs]
    ]

    # attn_0_3 ####################################################
    def predicate_0_3(q_position, k_position):
        if q_position in {0, 4}:
            return k_position == 3
        elif q_position in {1, 7}:
            return k_position == 4
        elif q_position in {2}:
            return k_position == 1
        elif q_position in {3, 5}:
            return k_position == 2
        elif q_position in {6}:
            return k_position == 5

    attn_0_3_pattern = select_closest(positions, positions, predicate_0_3)
    attn_0_3_outputs = aggregate(attn_0_3_pattern, tokens)
    attn_0_3_output_scores = classifier_weights.loc[
        [("attn_0_3_outputs", str(v)) for v in attn_0_3_outputs]
    ]

    # mlp_0_0 #####################################################
    def mlp_0_0(attn_0_2_output, position):
        key = (attn_0_2_output, position)
        if key in {
            ("0", 1),
            ("0", 2),
            ("0", 3),
            ("0", 4),
            ("1", 1),
            ("1", 2),
            ("2", 1),
            ("2", 2),
            ("3", 1),
            ("4", 1),
            ("<s>", 1),
        }:
            return 0
        elif key in {
            ("0", 0),
            ("0", 6),
            ("1", 0),
            ("1", 6),
            ("</s>", 0),
            ("</s>", 2),
            ("</s>", 3),
            ("</s>", 5),
            ("</s>", 6),
            ("</s>", 7),
        }:
            return 2
        elif key in {
            ("0", 5),
            ("0", 7),
            ("1", 3),
            ("1", 4),
            ("1", 5),
            ("1", 7),
            ("3", 2),
            ("4", 2),
            ("</s>", 1),
        }:
            return 7
        elif key in {
            ("3", 4),
            ("3", 5),
            ("3", 7),
            ("4", 4),
            ("4", 5),
            ("4", 7),
            ("<s>", 4),
        }:
            return 1
        elif key in {("2", 3), ("2", 4), ("2", 5), ("2", 7), ("3", 3), ("4", 3)}:
            return 6
        return 4

    mlp_0_0_outputs = [mlp_0_0(k0, k1) for k0, k1 in zip(attn_0_2_outputs, positions)]
    mlp_0_0_output_scores = classifier_weights.loc[
        [("mlp_0_0_outputs", str(v)) for v in mlp_0_0_outputs]
    ]

    # mlp_0_1 #####################################################
    def mlp_0_1(position):
        key = position
        if key in {1, 2}:
            return 7
        elif key in {5}:
            return 1
        return 6

    mlp_0_1_outputs = [mlp_0_1(k0) for k0 in positions]
    mlp_0_1_output_scores = classifier_weights.loc[
        [("mlp_0_1_outputs", str(v)) for v in mlp_0_1_outputs]
    ]

    # mlp_0_2 #####################################################
    def mlp_0_2(attn_0_1_output, position):
        key = (attn_0_1_output, position)
        if key in {
            ("1", 4),
            ("2", 2),
            ("2", 3),
            ("2", 4),
            ("2", 7),
            ("3", 2),
            ("3", 3),
            ("3", 4),
            ("3", 7),
            ("4", 2),
            ("4", 3),
            ("4", 4),
            ("4", 7),
        }:
            return 3
        elif key in {
            ("0", 0),
            ("0", 1),
            ("0", 2),
            ("0", 3),
            ("0", 4),
            ("0", 7),
            ("1", 1),
            ("2", 1),
            ("3", 1),
            ("4", 1),
            ("<s>", 1),
        }:
            return 6
        elif key in {
            ("0", 5),
            ("1", 2),
            ("1", 3),
            ("1", 7),
            ("</s>", 1),
            ("</s>", 2),
            ("</s>", 3),
            ("</s>", 7),
        }:
            return 5
        return 1

    mlp_0_2_outputs = [mlp_0_2(k0, k1) for k0, k1 in zip(attn_0_1_outputs, positions)]
    mlp_0_2_output_scores = classifier_weights.loc[
        [("mlp_0_2_outputs", str(v)) for v in mlp_0_2_outputs]
    ]

    # mlp_0_3 #####################################################
    def mlp_0_3(attn_0_2_output, position):
        key = (attn_0_2_output, position)
        if key in {
            ("0", 4),
            ("0", 7),
            ("1", 4),
            ("2", 4),
            ("2", 7),
            ("3", 4),
            ("3", 7),
            ("4", 4),
            ("</s>", 0),
            ("</s>", 1),
            ("</s>", 2),
            ("</s>", 3),
            ("</s>", 4),
            ("</s>", 7),
            ("<s>", 0),
            ("<s>", 1),
            ("<s>", 2),
            ("<s>", 3),
            ("<s>", 4),
            ("<s>", 6),
            ("<s>", 7),
        }:
            return 0
        elif key in {
            ("0", 5),
            ("0", 6),
            ("1", 5),
            ("1", 6),
            ("1", 7),
            ("2", 5),
            ("2", 6),
            ("3", 5),
            ("3", 6),
            ("4", 5),
            ("4", 6),
            ("</s>", 5),
            ("</s>", 6),
            ("<s>", 5),
        }:
            return 7
        return 5

    mlp_0_3_outputs = [mlp_0_3(k0, k1) for k0, k1 in zip(attn_0_2_outputs, positions)]
    mlp_0_3_output_scores = classifier_weights.loc[
        [("mlp_0_3_outputs", str(v)) for v in mlp_0_3_outputs]
    ]

    # attn_1_0 ####################################################
    def predicate_1_0(position, mlp_0_1_output):
        if position in {0, 1, 2, 3}:
            return mlp_0_1_output == 1
        elif position in {4, 5, 7}:
            return mlp_0_1_output == 0
        elif position in {6}:
            return mlp_0_1_output == 3

    attn_1_0_pattern = select_closest(mlp_0_1_outputs, positions, predicate_1_0)
    attn_1_0_outputs = aggregate(attn_1_0_pattern, attn_0_2_outputs)
    attn_1_0_output_scores = classifier_weights.loc[
        [("attn_1_0_outputs", str(v)) for v in attn_1_0_outputs]
    ]

    # attn_1_1 ####################################################
    def predicate_1_1(q_position, k_position):
        if q_position in {0, 4, 5, 7}:
            return k_position == 1
        elif q_position in {1}:
            return k_position == 0
        elif q_position in {2}:
            return k_position == 5
        elif q_position in {3}:
            return k_position == 6
        elif q_position in {6}:
            return k_position == 7

    attn_1_1_pattern = select_closest(positions, positions, predicate_1_1)
    attn_1_1_outputs = aggregate(attn_1_1_pattern, attn_0_0_outputs)
    attn_1_1_output_scores = classifier_weights.loc[
        [("attn_1_1_outputs", str(v)) for v in attn_1_1_outputs]
    ]

    # attn_1_2 ####################################################
    def predicate_1_2(q_position, k_position):
        if q_position in {0, 3}:
            return k_position == 1
        elif q_position in {1}:
            return k_position == 5
        elif q_position in {2}:
            return k_position == 6
        elif q_position in {4, 7}:
            return k_position == 2
        elif q_position in {5, 6}:
            return k_position == 3

    attn_1_2_pattern = select_closest(positions, positions, predicate_1_2)
    attn_1_2_outputs = aggregate(attn_1_2_pattern, tokens)
    attn_1_2_output_scores = classifier_weights.loc[
        [("attn_1_2_outputs", str(v)) for v in attn_1_2_outputs]
    ]

    # attn_1_3 ####################################################
    def predicate_1_3(position, attn_0_1_output):
        if position in {0}:
            return attn_0_1_output == "1"
        elif position in {1, 2}:
            return attn_0_1_output == "0"
        elif position in {3}:
            return attn_0_1_output == "<s>"
        elif position in {4, 5, 6}:
            return attn_0_1_output == "4"
        elif position in {7}:
            return attn_0_1_output == "3"

    attn_1_3_pattern = select_closest(attn_0_1_outputs, positions, predicate_1_3)
    attn_1_3_outputs = aggregate(attn_1_3_pattern, attn_0_1_outputs)
    attn_1_3_output_scores = classifier_weights.loc[
        [("attn_1_3_outputs", str(v)) for v in attn_1_3_outputs]
    ]

    # mlp_1_0 #####################################################
    def mlp_1_0(position, attn_1_2_output):
        key = (position, attn_1_2_output)
        if key in {
            (0, "0"),
            (1, "0"),
            (2, "0"),
            (2, "1"),
            (2, "2"),
            (2, "3"),
            (2, "4"),
            (3, "0"),
        }:
            return 5
        elif key in {(1, "1"), (1, "2"), (1, "3"), (1, "4")}:
            return 1
        elif key in {(0, "2"), (4, "2"), (5, "2")}:
            return 4
        elif key in {(0, "1"), (3, "1"), (4, "0")}:
            return 6
        elif key in {(3, "2"), (4, "1"), (5, "1")}:
            return 7
        elif key in {(5, "0")}:
            return 2
        return 0

    mlp_1_0_outputs = [mlp_1_0(k0, k1) for k0, k1 in zip(positions, attn_1_2_outputs)]
    mlp_1_0_output_scores = classifier_weights.loc[
        [("mlp_1_0_outputs", str(v)) for v in mlp_1_0_outputs]
    ]

    # mlp_1_1 #####################################################
    def mlp_1_1(attn_1_3_output, attn_1_0_output):
        key = (attn_1_3_output, attn_1_0_output)
        if key in {
            ("3", "2"),
            ("3", "3"),
            ("3", "4"),
            ("3", "</s>"),
            ("3", "<s>"),
            ("4", "2"),
            ("4", "3"),
            ("4", "4"),
            ("4", "</s>"),
            ("4", "<s>"),
            ("</s>", "2"),
            ("</s>", "3"),
            ("</s>", "4"),
            ("</s>", "</s>"),
            ("</s>", "<s>"),
            ("<s>", "3"),
            ("<s>", "4"),
            ("<s>", "</s>"),
            ("<s>", "<s>"),
        }:
            return 4
        elif key in {("0", "<s>"), ("2", "<s>")}:
            return 6
        return 1

    mlp_1_1_outputs = [
        mlp_1_1(k0, k1) for k0, k1 in zip(attn_1_3_outputs, attn_1_0_outputs)
    ]
    mlp_1_1_output_scores = classifier_weights.loc[
        [("mlp_1_1_outputs", str(v)) for v in mlp_1_1_outputs]
    ]

    # mlp_1_2 #####################################################
    def mlp_1_2(attn_1_0_output, position):
        key = (attn_1_0_output, position)
        if key in {
            ("2", 4),
            ("2", 5),
            ("3", 4),
            ("3", 5),
            ("4", 0),
            ("4", 4),
            ("4", 5),
            ("</s>", 4),
            ("</s>", 5),
            ("<s>", 4),
            ("<s>", 5),
        }:
            return 0
        elif key in {
            ("0", 3),
            ("1", 3),
            ("2", 3),
            ("3", 0),
            ("3", 3),
            ("4", 3),
            ("</s>", 3),
            ("<s>", 3),
        }:
            return 7
        elif key in {
            ("2", 1),
            ("2", 7),
            ("3", 1),
            ("3", 7),
            ("4", 1),
            ("4", 7),
            ("</s>", 1),
            ("</s>", 7),
        }:
            return 1
        elif key in {("0", 4), ("0", 5), ("1", 4), ("1", 5)}:
            return 4
        elif key in {("2", 0), ("2", 6), ("3", 6), ("4", 6)}:
            return 5
        elif key in {("1", 1), ("1", 7), ("</s>", 6), ("<s>", 6)}:
            return 2
        elif key in {("4", 2), ("<s>", 0), ("<s>", 2), ("<s>", 7)}:
            return 6
        return 3

    mlp_1_2_outputs = [mlp_1_2(k0, k1) for k0, k1 in zip(attn_1_0_outputs, positions)]
    mlp_1_2_output_scores = classifier_weights.loc[
        [("mlp_1_2_outputs", str(v)) for v in mlp_1_2_outputs]
    ]

    # mlp_1_3 #####################################################
    def mlp_1_3(attn_1_0_output, attn_1_3_output):
        key = (attn_1_0_output, attn_1_3_output)
        if key in {
            ("0", "3"),
            ("0", "</s>"),
            ("1", "3"),
            ("2", "3"),
            ("3", "0"),
            ("3", "1"),
            ("3", "2"),
            ("3", "3"),
            ("3", "</s>"),
            ("</s>", "0"),
            ("</s>", "1"),
            ("</s>", "2"),
            ("</s>", "3"),
            ("</s>", "</s>"),
            ("<s>", "3"),
        }:
            return 5
        elif key in {("3", "4")}:
            return 2
        return 7

    mlp_1_3_outputs = [
        mlp_1_3(k0, k1) for k0, k1 in zip(attn_1_0_outputs, attn_1_3_outputs)
    ]
    mlp_1_3_output_scores = classifier_weights.loc[
        [("mlp_1_3_outputs", str(v)) for v in mlp_1_3_outputs]
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
                mlp_0_2_output_scores,
                mlp_0_3_output_scores,
                attn_1_0_output_scores,
                attn_1_1_output_scores,
                attn_1_2_output_scores,
                attn_1_3_output_scores,
                mlp_1_0_output_scores,
                mlp_1_1_output_scores,
                mlp_1_2_output_scores,
                mlp_1_3_output_scores,
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
        ["<s>", "0", "4", "1", "1", "4", "2", "</s>"],
        ["<pad>", "0", "1", "1", "2", "4", "4", "<pad>"],
    ),
    (
        ["<s>", "4", "4", "3", "0", "2", "</s>"],
        ["<pad>", "0", "2", "3", "4", "4", "<pad>"],
    ),
    (
        ["<s>", "3", "0", "2", "2", "3", "3", "</s>"],
        ["<pad>", "0", "2", "2", "3", "3", "3", "<pad>"],
    ),
    (
        ["<s>", "2", "4", "2", "0", "0", "3", "</s>"],
        ["<pad>", "0", "0", "2", "2", "3", "4", "<pad>"],
    ),
    (
        ["<s>", "0", "0", "2", "0", "2", "3", "</s>"],
        ["<pad>", "0", "0", "0", "2", "2", "3", "<pad>"],
    ),
    (
        ["<s>", "0", "1", "0", "4", "3", "</s>"],
        ["<pad>", "0", "0", "1", "3", "4", "<pad>"],
    ),
    (
        ["<s>", "4", "2", "1", "2", "4", "3", "</s>"],
        ["<pad>", "1", "2", "2", "3", "4", "4", "<pad>"],
    ),
    (["<s>", "4", "3", "</s>"], ["<pad>", "3", "4", "<pad>"]),
    (
        ["<s>", "1", "1", "0", "4", "2", "1", "</s>"],
        ["<pad>", "0", "1", "1", "1", "2", "4", "<pad>"],
    ),
    (
        ["<s>", "0", "1", "1", "3", "1", "</s>"],
        ["<pad>", "0", "1", "1", "1", "3", "<pad>"],
    ),
]
for x, y in examples:
    print(f"x: {x}")
    print(f"y: {y}")
    y_hat = run(x)
    print(f"y_hat: {y_hat}")
    print()
