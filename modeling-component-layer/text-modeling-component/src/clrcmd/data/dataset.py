import csv
import logging
from typing import Dict, List, Tuple

import torch
from torch import Tensor
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase, default_data_collator

logger = logging.getLogger(__name__)


class STSBenchmarkDataset(Dataset):
    def __init__(
        self, examples: List[Tuple[Tuple[str, str], float]], tokenizer: PreTrainedTokenizerBase
    ):
        self.examples = examples
        self.tokenizer = tokenizer

    def __getitem__(self, idx: int) -> Tuple[Dict[str, Tensor], Dict[str, Tensor], Tensor]:
        (text1, text2), score = self.examples[idx]
        text1 = self.tokenizer(
            text1, truncation=True, padding="max_length", max_length=128, return_tensors="pt"
        )
        text2 = self.tokenizer(
            text2, truncation=True, padding="max_length", max_length=128, return_tensors="pt"
        )
        text1 = {k: v[0] for k, v in text1.items()}
        text2 = {k: v[0] for k, v in text2.items()}
        return {"inputs1": text1, "inputs2": text2, "label": torch.tensor(score)}

    def __len__(self):
        return len(self.examples)


class NLIContrastiveLearningDataset(Dataset):
    def __init__(self, filepath: str, tokenizer: PreTrainedTokenizerBase):
        with open(filepath) as f:
            self.examples = [
                (row["sent0"], row["sent1"], row["hard_neg"]) for row in csv.DictReader(f)
            ]
        self.tokenizer = tokenizer

    def __getitem__(
        self, idx: int
    ) -> Tuple[Dict[str, Tensor], Dict[str, Tensor], Dict[str, Tensor]]:
        text1, text2, text_neg = self.examples[idx]
        text1 = self.tokenizer(
            text1, truncation=True, padding="max_length", max_length=32, return_tensors="pt"
        )
        text2 = self.tokenizer(
            text2, truncation=True, padding="max_length", max_length=32, return_tensors="pt"
        )
        text_neg = self.tokenizer(
            text_neg, truncation=True, padding="max_length", max_length=32, return_tensors="pt"
        )
        text1 = {k: v[0] for k, v in text1.items()}
        text2 = {k: v[0] for k, v in text2.items()}
        text_neg = {k: v[0] for k, v in text_neg.items()}
        return {"inputs1": text1, "inputs2": text2, "inputs_neg": text_neg}

    def __len__(self) -> int:
        return len(self.examples)


class ContrastiveLearningCollator:
    def __call__(
        self, features: List[Tuple[Dict[str, Tensor], Dict[str, Tensor], Dict[str, Tensor]]]
    ) -> Tuple[Dict[str, Tensor], Dict[str, Tensor], Dict[str, Tensor]]:
        result = {}
        for k in features[0]:
            if k == "inputs1":
                result["inputs1"] = default_data_collator([x["inputs1"] for x in features])
            elif k == "inputs2":
                result["inputs2"] = default_data_collator([x["inputs2"] for x in features])
            elif k == "inputs_neg":
                result["inputs_neg"] = default_data_collator([x["inputs_neg"] for x in features])
            elif k == "label":
                result["label"] = torch.stack([x["label"] for x in features])
        return result
