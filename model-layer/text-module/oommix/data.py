import csv
import logging
import os
import subprocess
import sys
from collections import Counter, defaultdict

import nltk
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from filelock import FileLock
from nltk.tokenize import sent_tokenize
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from tqdm import tqdm

nltk.download("punkt")


class ListDataset(Dataset):
    def __init__(self, data, n_class):
        self.data = data
        self.n_class = n_class

    def __getitem__(self, idx):
        data = self.data[idx]
        return {
            "idx": idx,
            "input": torch.tensor(data["input"], dtype=torch.long),
            "label": torch.tensor(data["label"], dtype=torch.long),
        }

    def __len__(self):
        return len(self.data)


def load_csv(filepath, fieldnames=None):
    with open(filepath, newline="", encoding="UTF8") as f:
        reader = csv.DictReader(f, fieldnames=fieldnames)
        for row in reader:
            yield row


def save_csv(filepath, data, fieldnames):
    with open(filepath, "w", newline="", encoding="UTF8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)


def load_ag_news(filepath):
    return [
        {"label": int(row["class"]) - 1, "input": row["description"]}
        for row in tqdm(
            load_csv(filepath, ["class", "title", "description"]),
            desc="Load ag news dataset",
        )
    ]


def load_yahoo_answer(filepath):
    return [
        {
            "label": int(row["class"]) - 1,
            "input": row["title"] + " " + row["content"] + " " + row["answer"],
        }
        for row in tqdm(
            load_csv(filepath, ["class", "title", "content", "answer"]),
            desc="Load yahoo dataset",
        )
    ]


def load_amazon_review_polarity(filepath):
    return [
        {"label": int(row["class"]) - 1, "input": row["text"]}
        for row in tqdm(
            load_csv(filepath, ["class", "title", "text"]),
            desc="Load amazon review polarity dataset",
        )
    ]


def load_dbpedia(filepath):
    data = [
        {"label": int(row["class"]) - 1, "input": row["content"]}
        for row in tqdm(
            load_csv(filepath, ["class", "title", "content"]), desc="Load dbpedia"
        )
    ]
    return data


def create_metadata(dataset):
    if dataset == "ag_news":
        data_load_func = load_ag_news
        n_class = 4
        num_valid_data = 1900 * n_class
    elif dataset == "yahoo_answer":
        data_load_func = load_yahoo_answer
        n_class = 10
        num_valid_data = 5000 * n_class
        # num_valid_data = 2000 * n_class
    elif dataset == "amazon_review_polarity":
        data_load_func = load_amazon_review_polarity
        n_class = 2
        num_valid_data = 4000 * n_class
    elif dataset == "dbpedia":
        data_load_func = load_dbpedia
        n_class = 14
        num_valid_data = 2000 * n_class
    else:
        raise AttributeError("Invalid dataset")
    return data_load_func, n_class, num_valid_data


CACHE_DIR = "/data/sh0416/cache"


def split_train_validation(
    load_f,
    src_path: str,
    tgt_train_path: str,
    tgt_valid_path: str,
    num_train_data: int,
    num_valid_data: int,
) -> None:
    with FileLock(tgt_train_path + ".lock") as tgt_train_lock, FileLock(
        tgt_valid_path + ".lock"
    ) as tgt_valid_lock:
        if not os.path.exists(tgt_train_path) or not os.path.exists(tgt_valid_path):
            train_data = load_f(src_path)
            train_data, valid_data = train_test_split(
                train_data,
                test_size=num_valid_data,
                random_state=42,
                shuffle=True,
                stratify=[x["label"] for x in train_data],
            )
            # Sample training data
            if num_train_data != -1:
                _, train_data = train_test_split(
                    train_data,
                    test_size=num_train_data,
                    random_state=42,
                    shuffle=True,
                    stratify=[x["label"] for x in train_data],
                )
            # For valid data, sort by length to accelerate inference
            valid_data = sorted(valid_data, key=lambda x: len(x["input"]), reverse=True)
            save_csv(tgt_train_path, train_data, ["input", "label"])
            save_csv(tgt_valid_path, valid_data, ["input", "label"])


def apply_eda(train_data):
    train_df = pd.DataFrame(data=train_data)
    train_df.to_csv(
        "eda_input.tsv", columns=["label", "input"], sep="\t", index=False, header=False
    )
    subprocess.run(
        [
            "python",
            "eda_nlp/code/augment.py",
            "--num_aug",
            "1",
            "--input",
            "eda_input.tsv",
            "--output",
            "eda_output.tsv",
        ]
    )
    train_df = pd.read_csv("eda_output.tsv", names=["label", "input"], sep="\t")
    train_data = [
        {"input": row["input"], "label": row["label"]} for _, row in train_df.iterrows()
    ]
    os.remove("eda_input.tsv")
    os.remove("eda_output.tsv")
    return train_data


def apply_backtranslate(data):
    # Pretrained translation model
    en2ru = torch.hub.load(
        "pytorch/fairseq",
        "transformer.wmt19.en-ru.single_model",
        tokenizer="moses",
        bpe="fastbpe",
    ).cuda()
    ru2en = torch.hub.load(
        "pytorch/fairseq",
        "transformer.wmt19.ru-en.single_model",
        tokenizer="moses",
        bpe="fastbpe",
    ).cuda()

    result = list(data)
    sentences = [
        (idx, idx2, s)
        for idx, row in enumerate(data)
        for idx2, s in enumerate(sent_tokenize(row["input"]))
    ]
    sentences = sorted(sentences, key=lambda x: len(x[2]), reverse=True)
    with torch.no_grad():
        augmented_sentences = []
        for idx in tqdm(range(0, len(sentences), 32), desc="backtranslate"):
            batch = sentences[idx : min(idx + 32, len(sentences))]
            inputs = [s[2][:1024] if len(s[2]) > 1024 else s[2] for s in batch]
            middle = en2ru.translate(inputs, beam=5)
            middle = [s[:1024] if len(s) > 1024 else s for s in middle]
            new_inputs = ru2en.translate(middle, sampling=True)
            augmented_sentences.extend(new_inputs)
        augmented_sentences = [
            (idx, idx2, s) for (idx, idx2, _), s in zip(sentences, augmented_sentences)
        ]
        augmented_sentences = sorted(augmented_sentences)
        augmented_data = defaultdict(dict)
        for idx, idx2, s in augmented_sentences:
            augmented_data[idx][idx2] = s
        augmented_data = {
            k: " ".join([s for _, s in sorted(v.items())])
            for k, v in augmented_data.items()
        }
        result.extend(
            [{"input": v, "label": data[k]["label"]} for k, v in augmented_data.items()]
        )
    return result


def apply_ssmba(train_data):
    with FileLock("ssmba.lock") as lock:
        with open("ssmba_input", "w") as f, open("ssmba_label", "w") as f2:
            for row in train_data:
                f.write(row["input"] + "\n")
                f2.write(str(row["label"]) + "\n")
        subprocess.run(
            [
                "python",
                "ssmba/ssmba.py",
                "--model",
                "bert-base-uncased",
                "--in-file",
                "ssmba_input",
                "--label-file",
                "ssmba_label",
                "--output-prefix",
                "ssmba_output",
                "--noise-prob",
                "0.25",
                "--num-samples",
                "1",
            ]
        )
        with open("ssmba_output", "r") as f, open("ssmba_output.label", "r") as f2:
            for inputs, labels in zip(f, f2):
                train_data.append({"input": inputs, "label": int(labels)})
    os.remove("ssmba_input")
    os.remove("ssmba_label")
    os.remove("ssmba_output")
    os.remove("ssmba_output.label")
    os.remove("ssmba.lock")
    return train_data


def apply_augmentation(src_path: str, tgt_path: str, augmentation: str) -> None:
    with FileLock(tgt_path + ".lock") as lock:
        if not os.path.exists(tgt_path):
            data = list(load_csv(src_path))
            if augmentation == "none":
                pass
            elif augmentation == "eda":
                data = apply_eda(data)
            elif augmentation == "backtranslate":
                data = apply_backtranslate(data)
            elif augmentation == "ssmba":
                data = apply_ssmba(data)
            else:
                raise AttributeError()
            # Overwrite existing training data to augmented one
            save_csv(tgt_path, data, ["input", "label"])
        else:
            data = list(load_csv(tgt_path))
    return data


def tokenize(load_f, src_path, tgt_path, tokenizer):
    with FileLock(tgt_path + ".lock") as lock:
        if not os.path.exists(tgt_path):
            data = list(load_f(src_path))
            for row in tqdm(data, desc="Tokenization"):
                row["input"] = " ".join(
                    map(
                        str,
                        tokenizer(row["input"], max_length=256, truncation=True)[
                            "input_ids"
                        ],
                    )
                )
            save_csv(tgt_path, data, ["input", "label"])
        else:
            data = load_csv(tgt_path)
    return [
        {"input": list(map(int, row["input"].split(" "))), "label": int(row["label"])}
        for row in data
    ]


def create_train_and_valid_dataset(
    dataset,
    dirpath,
    augmentation="none",
    tokenizer=None,
    num_train_data=-1,
    return_type="pytorch",
):
    """Create dataset for training script or analyzing data.

    :param dataset: The name of dataset
    :type dataset: str
    :param dirpath: The directory path for actual raw data
    :type dirpath: str
    :param tokenizer: The tokenizer to tokenize real text into sequence of tokens
    :type tokenizer: huggingface transformer package Tokenizer class
    :param num_train_data: The number of available training data instances, defaults to -1 means the whole data
    :type num_train_data: int, optional
    :param return_type: The returned type, if "pytorch" means the data is represented as Dataset,
                         if "pandas" means the data is represented as DataFrame, defaults to "pytorch"
    :type return_type: str, optional
    :return: [description]
    :rtype: [type]
    """
    data_load_func, n_class, num_valid_data = create_metadata(dataset)
    cache_path = "%s_%d_%s" % (dataset, num_train_data, augmentation)
    # 0. Make cache directory
    os.makedirs(os.path.join(CACHE_DIR, cache_path), exist_ok=True)
    # 1. Train Validation split
    split_train_validation(
        data_load_func,
        os.path.join(dirpath, "train.csv"),
        os.path.join(CACHE_DIR, cache_path, "train.csv"),
        os.path.join(CACHE_DIR, cache_path, "valid.csv"),
        num_train_data,
        num_valid_data,
    )
    # 2. Apply augmentation if specified
    apply_augmentation(
        os.path.join(CACHE_DIR, cache_path, "train.csv"),
        os.path.join(CACHE_DIR, cache_path, "train_augmented.csv"),
        augmentation,
    )
    # 3. Tokenize given data
    train_data = tokenize(
        load_csv,
        os.path.join(CACHE_DIR, cache_path, "train_augmented.csv"),
        os.path.join(CACHE_DIR, cache_path, "train_augmented_tokenized.csv"),
        tokenizer,
    )
    valid_data = tokenize(
        load_csv,
        os.path.join(CACHE_DIR, cache_path, "valid.csv"),
        os.path.join(CACHE_DIR, cache_path, "valid_tokenized.csv"),
        tokenizer,
    )
    # Statistics: Average length
    length = [len(row["input"]) for row in train_data]
    logging.info("Train data average length: %.2f" % (sum(length) / len(length)))
    length = [len(row["input"]) for row in valid_data]
    logging.info("Valid data average length: %.2f" % (sum(length) / len(length)))

    # Calculate the observed token number
    train_token = set(token for row in train_data for token in row["input"])
    valid_token = set(token for row in valid_data for token in row["input"])
    oov_token = valid_token - train_token
    logging.info("Train observed token number: %d" % len(train_token))
    logging.info("Valid observed token number: %d" % len(valid_token))
    logging.info("Out of vocabulary token number: %d" % len(oov_token))
    logging.info("Ouf of vocabulary rate: %.4f" % (len(oov_token) / len(valid_token)))
    if return_type == "pytorch":
        train_dataset = ListDataset(train_data, n_class)
        valid_dataset = ListDataset(valid_data, n_class)
    elif return_type == "pandas":
        train_dataset = pd.DataFrame(data=train_data)
        valid_dataset = pd.DataFrame(data=valid_data)
    return train_dataset, valid_dataset


def create_test_dataset(dataset, dirpath, tokenizer=None, return_type="pytorch"):
    data_load_func, n_class, _ = create_metadata(dataset)
    cache_path = "%s" % (dataset)
    # 0. Make cache directory
    os.makedirs(os.path.join(CACHE_DIR, cache_path), exist_ok=True)
    # 1. Tokenize given data
    test_data = tokenize(
        data_load_func,
        os.path.join(dirpath, "test.csv"),
        os.path.join(CACHE_DIR, cache_path, "test_tokenized.csv"),
        tokenizer,
    )
    test_data = sorted(test_data, key=lambda x: len(x["input"]), reverse=True)
    # Statistics: Average length
    length = [len(row["input"]) for row in test_data]
    logging.info("Test data average length: %.2f" % (sum(length) / len(length)))
    if return_type == "pytorch":
        test_dataset = ListDataset(test_data, n_class)
    elif return_type == "pandas":
        test_dataset = pd.DataFrame(data=test_data)
    return test_dataset


class CollateFn:
    def __init__(self, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.max_length = max_length
        logging.info(
            "Special token %s: %d"
            % (self.tokenizer.cls_token, self.tokenizer.cls_token_id)
        )
        logging.info(
            "Special token %s: %d"
            % (self.tokenizer.sep_token, self.tokenizer.sep_token_id)
        )
        logging.info(
            "Special token %s: %d"
            % (self.tokenizer.pad_token, self.tokenizer.pad_token_id)
        )
        logging.info("Max length: %d" % max_length)

    def __call__(self, batch):
        inputs = {}
        with torch.no_grad():
            idx = torch.tensor([x["idx"] for x in batch], dtype=torch.long)
            inputs["input_ids"] = nn.utils.rnn.pad_sequence(
                [x["input"] for x in batch],
                batch_first=True,
                padding_value=self.tokenizer.pad_token_id,
            )
            if inputs["input_ids"].shape[1] > self.max_length:
                inputs["input_ids"] = inputs["input_ids"][:, : self.max_length]
            inputs["attention_mask"] = (
                inputs["input_ids"] != self.tokenizer.pad_token_id
            )
            """
            inputs["mixup_mask"] = {
                    "is_cls": (inputs["input_ids"] == self.tokenizer.cls_token_id),
                    "is_sep": (inputs["input_ids"] == self.tokenizer.sep_token_id),
                    "is_normal": ((inputs["input_ids"] != self.tokenizer.cls_token_id) & \
                                  (inputs["input_ids"] != self.tokenizer.sep_token_id) & \
                                  (inputs["input_ids"] != self.tokenizer.pad_token_id))}
            """
            labels = torch.stack([x["label"] for x in batch])
        return {"idx": idx, "inputs": inputs, "labels": labels}
