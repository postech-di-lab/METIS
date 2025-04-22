import random

import pandas as pd
import torch
from loguru import logger
from tqdm import tqdm

from humanextension.utils import remove_after_stop_token


def create_prompt_for_target_function_implementation_humanextension(
        row: dict) -> str:
    return f"{row['imports']}\n\n\n{row['function1_signature']}{row['function1_implementation']}\n\n{row['function2_signature']}".strip(
    )


def implement_step_by_step_humanextension(
    df: pd.DataFrame,
    generate_fn: callable,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    num_implementations: int,
    seed: int,
) -> list[list[tuple[str, list[str]]]]:
    torch.manual_seed(seed)

    total_implementations = []
    for _, row in tqdm(df.iterrows(), total=df.shape[0],
                       desc="Implementation"):
        # Create prompt for implementing auxiliary function
        prompt = f"{row['imports']}\n{row['function1_signature']}".strip()
        stop_tokens = row["stop_tokens"]

        status_list: list[list[str]] = []
        auxiliary_functions: list[str] = []
        while len(auxiliary_functions) < num_implementations:
            # Generate auxiliary function implementation
            completions = generate_fn(
                [prompt],
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                stop_sequences=stop_tokens,
                num_return_sequences=4,
            )[0]

            # Parse completion to extract implementation
            for text, finish_reason in completions:
                implementation = remove_after_stop_token(
                    text, prompt, stop_tokens)
                if finish_reason != "length":
                    status = "ok"
                else:
                    status = "max generation token reach in auxiliary function implementation"
                auxiliary_functions.append(implementation)
                status_list.append([status])

        # Truncate implementation if the number of auxiliary function is larget than given setting
        auxiliary_functions = auxiliary_functions[:num_implementations]
        status_list = status_list[:num_implementations]

        # Create prompt for implementing target function with generated auxiliary function
        prompts = [
            f"{auxiliary_function}\n{row['function2_signature']}".strip()
            for auxiliary_function in auxiliary_functions
        ]

        target_functions: list[str] = []
        for prompt_idx in range(0, len(prompts), 4):
            batch_prompts = prompts[prompt_idx:prompt_idx + 4]
            # Generate target function implementation
            batch_completions = generate_fn(
                batch_prompts,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                stop_sequences=stop_tokens,
                num_return_sequences=1,
            )
            batch_completions = [(completion, finish_reason)
                                 for element in batch_completions
                                 for completion, finish_reason in element]

            # Parse completion to extract implementation
            for idx, (prompt, (text, finish_reason)) in enumerate(
                    zip(batch_prompts, batch_completions)):
                implementation = remove_after_stop_token(
                    text, prompt, stop_tokens)
                if finish_reason != "length":
                    status = "ok"
                else:
                    status = "max generation token reach in target function implementation"
                target_functions.append(implementation)
                status_list[prompt_idx + idx].append(status)

        assert len(target_functions) == len(status_list) == num_implementations
        total_implementations.append(list(zip(target_functions, status_list)))

    return total_implementations
