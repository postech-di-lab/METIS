import argparse
import csv
import itertools
from collections import defaultdict
from typing import Callable, Dict, List, NamedTuple, Tuple

import matplotlib.pyplot as plt
import torch
from torch import Tensor
from transformers import RobertaModel, RobertaTokenizer


class TextRepresentation(NamedTuple):
    tokens: List[str]
    # Alignment representation (consistent with byte representation)
    bytes: List[str]  # Byte representation
    string: List[str]  # Unicode representation


def create_create_representation(tokenizer: RobertaTokenizer) -> Callable:
    def create_representation(text: str) -> TextRepresentation:
        tokens = text.split()
        bytes = "".join(tokens)
        string = tokenizer.convert_tokens_to_string(bytes)
        return TextRepresentation(tokens=tokens, bytes=bytes, string=string)

    return create_representation


Pair = Tuple[int, int]


def create_alignment(stream: List[str], tokens: List[str]) -> List[Pair]:
    assert "".join(tokens) == stream
    start_pos = itertools.accumulate(map(len, tokens), initial=0)
    length = map(len, tokens)
    return list(map(lambda x: (x[0], x[0] + x[1]), zip(start_pos, length)))


def is_overlap(interval1: Pair, interval2: Pair) -> bool:
    assert interval1[0] < interval1[1]
    assert interval2[0] < interval2[1]
    state = sorted(
        [
            (interval1[0], 0),
            (interval1[1], 1),
            (interval2[0], 0),
            (interval2[1], 1),
        ],
        key=lambda x: (x[0], 1 - x[1]),
    )
    state = list(map(lambda x: x[1], state))
    # In this case, only two possible cases are yield in this logic,
    # (0, 0, 1, 1), which is overlapped, or (0, 1, 0, 1), which is exclusive
    return state == [0, 0, 1, 1]


def create_overlap_pair(align1: List[Pair], align2: List[Pair]) -> List[Tuple[Pair, Pair]]:
    pipeline = itertools.product(align1, align2)
    pipeline = filter(lambda x: is_overlap(x[0], x[1]), pipeline)
    return list(pipeline)


def create_perfect_overlap_pair(align1: List[Pair], align2: List[Pair]) -> List[Tuple[Pair, Pair]]:
    pipeline = itertools.product(align1, align2)
    pipeline = filter(lambda x: x[0] == x[1], pipeline)
    return list(pipeline)


def get_first_diff_position(align1: List[Pair], align2: List[Pair]) -> Tuple[Pair, Pair]:
    x = zip(align1, align2)
    x = itertools.dropwhile(lambda x: x[0] == x[1], x)
    x = itertools.islice(x, 1)
    return list(x)[0]


def index_pair_list(
    overlap_pair: List[Tuple[Pair, Pair]],
    pair2idx1: Dict[Pair, int],
    pair2idx2: Dict[Pair, int],
) -> List[Tuple[int, int]]:
    x = map(lambda x: index_pair(x, pair2idx1, pair2idx2), overlap_pair)
    return list(x)


def index_pair(
    pair: Tuple[Pair, Pair],
    pair2idx1: Dict[Pair, int],
    pair2idx2: Dict[Pair, int],
) -> Tuple[int, int]:
    return (pair2idx1[pair[0]], pair2idx2[pair[1]])


def distance(x: Tensor, y: Tensor) -> float:
    return torch.pow(x - y, 2).sum().item()


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    "--data-path",
    type=str,
    default=".data/wiki1m_for_simcse.txt_bpedropout_0.01_roberta-base.csv",
    help="data path",
)
parser.add_argument("--data-size", type=int, default=1000, help="data size")
parser.add_argument("--checkpoint", type=str, default="roberta-base", help="checkpoint")


def main(args: argparse.Namespace):
    with open(args.data_path, newline="") as f:
        reader = csv.DictReader(f)
        reader = filter(lambda x: x["input_strs"] != x["input_strs2"], reader)
        reader = itertools.islice(reader, args.data_size)
        data = list(reader)

    tokenizer = RobertaTokenizer.from_pretrained(args.checkpoint)

    create_representation = create_create_representation(tokenizer)
    text1 = list(map(lambda x: create_representation(x["input_strs"]), data))
    text2 = list(map(lambda x: create_representation(x["input_strs2"]), data))
    align1 = list(map(lambda x: create_alignment(x.bytes, x.tokens), text1))
    align2 = list(map(lambda x: create_alignment(x.bytes, x.tokens), text2))
    interval2idx1 = [{x: i for i, x in enumerate(a)} for a in align1]
    interval2idx2 = [{x: i for i, x in enumerate(a)} for a in align2]

    overlap_perfect = list(map(create_perfect_overlap_pair, align1, align2))
    first_diff = list(map(get_first_diff_position, align1, align2))

    overlap_perfect_idx = list(map(index_pair_list, overlap_perfect, interval2idx1, interval2idx2))
    first_diff_idx = list(map(index_pair, first_diff, interval2idx1, interval2idx2))

    model = RobertaModel.from_pretrained(
        "./result/my-unsup-simcse-roberta-base-0.0-0.00001-32-42/checkpoint-1000"
    )
    model.eval()

    token_diff_dict = defaultdict(list)
    for x in zip(text1, text2, overlap_perfect_idx, first_diff_idx):
        x1, x2, overlap_perfect_idxes, first_diff_pos = x
        if first_diff_pos[0] > 32:
            continue
        sent_features = tokenizer.batch_encode_plus(
            [
                tokenizer.convert_tokens_to_ids(x1.tokens),
                tokenizer.convert_tokens_to_ids(x2.tokens),
            ],
            is_split_into_words=True,
            max_length=32,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        outputs = model(**sent_features)

        # Due to the special ID [CLS], idx is right shifted by one
        overlap_perfect_idxes = [(x[0] + 1, x[1] + 1) for x in overlap_perfect_idxes]
        overlap_perfect_idxes = list(
            filter(lambda x: x[0] < 32 and x[1] < 32, overlap_perfect_idxes)
        )
        token_diff = [
            distance(
                outputs.last_hidden_state[0, i],
                outputs.last_hidden_state[1, j],
            )
            for i, j in overlap_perfect_idxes
        ]
        print(f"{token_diff = }")
        print(f"{first_diff_pos = }")
        for i, d in enumerate(token_diff):
            token_diff_dict[i - first_diff_pos[0] + 1].append(d)

    fig, ax = plt.subplots(figsize=(20, 8), constrained_layout=True)
    token_diff_dict = token_diff_dict.items()
    # token_diff_dict = list(filter(lambda x: x[0] < 30, token_diff_dict))
    token_diff_dict = sorted(token_diff_dict)
    labels = [x[0] for x in token_diff_dict]
    values = [x[1] for x in token_diff_dict]
    ax.boxplot(values, labels=labels)
    ax.tick_params(axis="x", labelsize=13, rotation=90)
    ax.tick_params(axis="y", labelsize=13)
    plt.savefig("analysis-token-diff-relative-pos.png")


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
