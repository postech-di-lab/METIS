import logging
import re
from typing import List, Optional, Tuple, TypedDict

import numpy as np
import torch
from bs4 import BeautifulSoup
from tokenizations import get_alignments
from transformers import PreTrainedTokenizerBase

from clrcmd.models import ModelInput, SentenceSimilarityModel

logger = logging.getLogger(__name__)


class AlignmentPair(TypedDict):
    sent1_word_ids: List[int]  # indexed by word tokenization
    sent2_word_ids: List[int]  # indexed by word tokenization
    type: str
    score: Optional[float]
    comment: str


class Alignment(TypedDict):
    id: int
    sent1: str  # word tokenized sentence with space
    sent2: str  # word tokenized sentence with space
    pairs: List[AlignmentPair]


def load_alignment(filepath: str) -> List[Alignment]:
    data = []
    with open(filepath) as f:
        soup = BeautifulSoup(f.read())
    for sentence in soup.find_all("sentence"):
        example_id = int(sentence.get("id"))
        sent1, sent2 = sentence.find(text=True).strip().splitlines()
        assert sent1.startswith("// ") and sent2.startswith("// ")
        sent1, sent2 = sent1[3:], sent2[3:]
        pairs = []
        for x in sentence.find("alignment").find(text=True).strip().splitlines():
            pair_ids, type, score, comment = x.split(" // ")
            sent1_word_ids, sent2_word_ids = pair_ids.split(" <==> ")
            sent1_word_ids = [int(x) for x in sent1_word_ids.split()]
            sent2_word_ids = [int(x) for x in sent2_word_ids.split()]
            score = None if score == "NIL" else float(score)
            pairs.append(
                {
                    "sent1_word_ids": sent1_word_ids,
                    "sent2_word_ids": sent2_word_ids,
                    "type": type,
                    "score": score,
                    "comment": comment,
                }
            )
        data.append({"id": example_id, "sent1": sent1, "sent2": sent2, "pairs": pairs})
    return data


def save_alignment(data: List[Alignment], filepath: str):
    """Save alignments with formatted text"""
    with open(filepath, "w") as f:
        for example in data:
            f.write(f"<sentence id=\"{example['id']}\" status=\"\">\n")
            f.write(f"// {example['sent1']}\n")
            f.write(f"// {example['sent2']}\n")
            f.write("<source>\n")
            for id, word in enumerate(example["sent1"].split(), start=1):
                f.write(f"{id} {word} :\n")
            f.write("</source>\n")
            f.write("<translation>\n")
            for id, word in enumerate(example["sent2"].split(), start=1):
                f.write(f"{id} {word} :\n")
            f.write("</translation>\n")
            f.write("<alignment>\n")
            for a in example["pairs"]:
                str_sent1_word_ids = " ".join(map(str, a["sent1_word_ids"]))
                str_sent2_word_ids = " ".join(map(str, a["sent2_word_ids"]))
                str_score = "NIL" if a["score"] is None else str(a["score"])
                f.write(f"{str_sent1_word_ids} <==> {str_sent2_word_ids} // ")
                f.write(f"{a['type']} // {str_score} // {a['comment']}\n")
            f.write("</alignment>\n")
            f.write("</sentence>\n\n\n")


class Example(TypedDict):
    id: int
    sent1: str
    sent2: str
    sent1_chunk: List[str]  # gold standard chunk sentence
    sent2_chunk: List[str]  # gold standard chunk sentence


def load_examples(
    filepath_sent1: str,
    filepath_sent2: str,
    filepath_sent1_chunk: str,
    filepath_sent2_chunk: str,
) -> List[Example]:
    with open(filepath_sent1) as f:
        sent1 = [x.strip() for x in f]
    with open(filepath_sent2) as f:
        sent2 = [x.strip() for x in f]
    with open(filepath_sent1_chunk) as f:
        sent1_chunk = [x.strip() for x in f]
    with open(filepath_sent2_chunk) as f:
        sent2_chunk = [x.strip() for x in f]
    pattern = re.compile(r"\[\s?(.*?)\s?\]")
    sent1_chunk = [pattern.findall(x) for x in sent1_chunk]
    sent2_chunk = [pattern.findall(x) for x in sent2_chunk]
    assert len(sent1) == len(sent2) == len(sent1_chunk) == len(sent2_chunk)
    for idx, (s1, s1_chunk) in enumerate(zip(sent1, sent1_chunk), start=1):
        assert s1 == " ".join(s1_chunk), f"{idx = } {s1 = } {s1_chunk = }"
    for idx, (s2, s2_chunk) in enumerate(zip(sent2, sent2_chunk), start=1):
        assert s2 == " ".join(s2_chunk), f"{idx = } {s2 = } {s2_chunk = }"
    return [
        {
            "id": idx,
            "sent1": s1,
            "sent2": s2,
            "sent1_chunk": s1_chunk,
            "sent2_chunk": s2_chunk,
        }
        for idx, (s1, s2, s1_chunk, s2_chunk) in enumerate(
            zip(sent1, sent2, sent1_chunk, sent2_chunk), start=1
        )
    ]


class PreprocessedExample(TypedDict):
    instance: Example
    sent1_token: List[str]  # Subword tokenized sequence for model heatmap
    sent2_token: List[str]  # Subword tokenized sequence for model heatmap
    inputs1: ModelInput
    inputs2: ModelInput


def preprocess(
    tokenizer: PreTrainedTokenizerBase, examples: List[Example]
) -> List[PreprocessedExample]:
    def tokenize(s: str):
        return tokenizer.convert_ids_to_tokens(tokenizer(s, add_special_tokens=False)["input_ids"])

    prep_examples = []
    for example in examples:
        sent1_token = tokenize(example["sent1"])
        sent2_token = tokenize(example["sent2"])
        inputs1 = tokenizer(example["sent1"], return_tensors="pt")
        inputs2 = tokenizer(example["sent2"], return_tensors="pt")
        prep_examples.append(
            {
                "example": example,
                "sent1_token": sent1_token,
                "sent2_token": sent2_token,
                "inputs1": inputs1,
                "inputs2": inputs2,
            }
        )
    return prep_examples


class InferedExample(TypedDict):
    example: Example
    sent1_token: List[str]  # Subword tokenized sequence for model heatmap
    sent2_token: List[str]  # Subword tokenized sequence for model heatmap
    heatmap_token: np.ndarray
    heatmap_chunk: np.ndarray
    pairs: List[AlignmentPair]


def inference(
    model: SentenceSimilarityModel, prep_examples: List[PreprocessedExample], device=torch.device
) -> List[InferedExample]:
    model = model.to(device)
    model.eval()
    infered_examples = []
    for prep_example in prep_examples:
        logger.debug(f"{prep_example}")
        # Compute heatmap
        with torch.no_grad():
            inputs1 = prep_example["inputs1"].to(device)
            inputs2 = prep_example["inputs2"].to(device)
            heatmap_token = model.compute_heatmap(inputs1, inputs2).cpu()
            heatmap_token = heatmap_token[0, 1:-1, 1:-1].numpy()
        align_sent1_token2chunk = get_alignments(
            prep_example["sent1_token"], prep_example["example"]["sent1_chunk"]
        )
        align_sent2_token2chunk = get_alignments(
            prep_example["sent2_token"], prep_example["example"]["sent2_chunk"]
        )
        logger.debug(f"{align_sent1_token2chunk}")
        logger.debug(f"{align_sent2_token2chunk}")
        heatmap_chunk = pool_heatmap(
            heatmap_token, align_sent1_token2chunk, align_sent2_token2chunk
        )
        mask1 = heatmap_chunk == np.max(heatmap_chunk, axis=0, keepdims=True)
        mask2 = heatmap_chunk == np.max(heatmap_chunk, axis=1, keepdims=True)
        sent1_chunks, sent2_chunks = np.nonzero(mask1 & mask2)
        align_sent1_chunk2word, _ = get_alignments(
            prep_example["example"]["sent1_chunk"],
            prep_example["example"]["sent1"].split(),
        )
        align_sent2_chunk2word, _ = get_alignments(
            prep_example["example"]["sent2_chunk"],
            prep_example["example"]["sent2"].split(),
        )
        sent1_word_ids = [[y + 1 for y in align_sent1_chunk2word[x]] for x in sent1_chunks]
        sent2_word_ids = [[y + 1 for y in align_sent2_chunk2word[x]] for x in sent2_chunks]
        pairs = [
            {
                "sent1_word_ids": s1,
                "sent2_word_ids": s2,
                "type": "EQUI",
                "score": 5,
                "comment": "",
            }
            for s1, s2 in zip(sent1_word_ids, sent2_word_ids)
        ]
        infered_examples.append(
            {
                "example": prep_example["example"],
                "sent1_token": prep_example["sent1_token"],
                "sent2_token": prep_example["sent2_token"],
                "heatmap_token": heatmap_token,
                "heatmap_chunk": heatmap_chunk,
                "pairs": pairs,
            }
        )
    return infered_examples


def save(examples: List[InferedExample], filepath: str):
    alignments = []
    for example in examples:
        alignments.append(
            {
                "id": example["example"]["id"],
                "sent1": example["example"]["sent1"],
                "sent2": example["example"]["sent2"],
                "pairs": example["pairs"],
            }
        )
    save_alignment(alignments, filepath)


TokensPair = Tuple[List[str], List[str]]
Alignment = List[TokensPair]


def pool_heatmap(
    heatmap: np.ndarray,
    align_sent1: Tuple[List[List[int]], List[List[int]]],
    align_sent2: Tuple[List[List[int]], List[List[int]]],
) -> np.ndarray:
    sent1_chunk_len, sent2_chunk_len = len(align_sent1[1]), len(align_sent2[1])
    # Split heatmap blockwise

    heatmap_splitted = [[heatmap[np.ix_(x, y)] for y in align_sent2[1]] for x in align_sent1[1]]
    heatmap_pooled1 = np.zeros((sent1_chunk_len, sent2_chunk_len))
    heatmap_pooled2 = np.zeros((sent1_chunk_len, sent2_chunk_len))
    for i, x in enumerate(heatmap_splitted):
        for j, y in enumerate(x):
            heatmap_pooled1[i, j] = np.mean(np.mean(y, axis=1))
            heatmap_pooled2[i, j] = np.mean(np.mean(y, axis=0))
    heatmap_pooled = (heatmap_pooled1 + heatmap_pooled2) / 2
    return heatmap_pooled
