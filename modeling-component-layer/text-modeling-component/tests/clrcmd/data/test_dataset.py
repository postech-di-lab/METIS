from functools import partial

import pytest
from transformers import AutoTokenizer

from clrcmd.data.dataset import NLIContrastiveLearningDataset


@pytest.fixture
def corpus_nli():
    return [
        "sent0,sent1,hard_neg",
        "my name is seonghyeon,seonghyeon is my name,hyeonchul is my name",
        "how are you?,how do you feel?,i'm fine thank you and you?",
        "i'm fine thank you and you?,i'm great!,today is terrible.",
    ]


@pytest.fixture
def filepath_nli(tmpdir, corpus_nli):
    tmpfile = tmpdir.join("nli.csv")
    tmpfile.write("\n".join(corpus_nli))
    return tmpfile.strpath


@pytest.fixture
def tokenizer():
    return AutoTokenizer.from_pretrained("roberta-base", use_fast=False)


def test_nli_dataset(filepath_nli, tokenizer, corpus_nli):
    dataset = NLIContrastiveLearningDataset(filepath_nli, tokenizer)
    tokenize_fn = partial(
        tokenizer.__call__,
        padding="max_length",
        max_length=32,
        truncation=True,
    )
    for i in range(len(dataset)):
        raw = corpus_nli[i + 1]
        example = dataset[i]
        x, pos, neg = raw.split(",")
        assert {k: v.tolist() for k, v in example["inputs1"].items()} == tokenize_fn(x)
        assert {k: v.tolist() for k, v in example["inputs2"].items()} == tokenize_fn(pos)
        assert {k: v.tolist() for k, v in example["inputs_neg"].items()} == tokenize_fn(neg)
