import pandas as pd
import torch
from tqdm import tqdm

from humanextension.utils import remove_after_stop_token


def create_prompt_humaneval(row: dict) -> str:
    return row["prompt"].strip()


def create_prompt_humanextension(row: dict) -> str:
    return f"{row['imports']}\n\n\n{row['function2_signature']}".strip()


def _generate_implementations(prompt: str, generate_fn: callable, temperature: float, top_p: float, max_new_tokens: int, stop_sequences: list[str], num_implementations: int, num_return_sequences: int) -> list[tuple[str, str]]:
    implementations = []
    while len(implementations) < num_implementations:
        completions = generate_fn([prompt], max_new_tokens=max_new_tokens, temperature=temperature, top_p=top_p, stop_sequences=stop_sequences, num_return_sequences=num_return_sequences)[0]

        # Parse completion to extract implementation
        for text, finish_reason in completions:
            implementation = remove_after_stop_token(text, prompt, stop_sequences)
            implementations.append((implementation, finish_reason))

    # Truncate implemenetation if the number of implementation is larger than given setting
    implementations = implementations[:num_implementations]
    return implementations


def implement_direct_humaneval(df: pd.DataFrame, generate_fn: callable, temperature: float, top_p: float, max_new_tokens: int, num_implementations: int, num_return_sequences: int, seed: int) -> list[list[tuple[str, str]]]:
    torch.manual_seed(seed)
    prompts = df.apply(create_prompt_humaneval, axis=1)
    stop_tokens_list = df["stop_tokens"]
    return [
        _generate_implementations(prompt, generate_fn, temperature, top_p, max_new_tokens, stop_sequences, num_implementations, num_return_sequences)
        for prompt, stop_sequences in tqdm(list(zip(prompts, stop_tokens_list)), desc="Implementation")
    ]


def implement_direct_humanextension(df: pd.DataFrame, generate_fn: callable, temperature: float, top_p: float, max_new_tokens: int, num_implementations: int, num_return_sequences: int, seed: int) -> list[list[tuple[str, str]]]:
    torch.manual_seed(seed)
    prompts = df.apply(create_prompt_humanextension, axis=1)
    stop_tokens_list = df["stop_tokens"]
    return [
        _generate_implementations(prompt, generate_fn, temperature, top_p, max_new_tokens, stop_sequences, num_implementations, num_return_sequences)
        for prompt, stop_sequences in tqdm(list(zip(prompts, stop_tokens_list)), desc="Implementation")
    ]