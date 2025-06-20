import pandas as pd
from datasets import load_dataset


def load_humaneval_openai():
    dataset_hf = load_dataset("openai_humaneval", split="test")
    return pd.DataFrame.from_records(dataset_hf)


def load_humaneval_multiple():
    dataset_hf = load_dataset("nuprl/MultiPL-E", "humaneval-py", split="test")
    return pd.DataFrame.from_records(dataset_hf)


def load_humanextension() -> pd.DataFrame:
    dataset_hf = load_dataset("sh0416/humanextension", split="test")
    return pd.DataFrame.from_records(dataset_hf)
