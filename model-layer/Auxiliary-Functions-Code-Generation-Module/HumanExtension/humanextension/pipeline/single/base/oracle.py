import pandas as pd
import torch
import random
from loguru import logger
from tqdm import tqdm

from humanextension.utils import (
    remove_after_stop_token,
    replace_function_name,
    parse_docstring,
    replace_function_docstring,
    remove_function_docstring,
)


def implement_oracle_humanextension(
    df: pd.DataFrame,
    generate_fn: callable,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
    num_implementations: int,
    shuffle_auxiliary_function_name: bool,
    shuffle_auxiliary_function_docstring: bool,
    remove_auxiliary_function_docstring: bool,
    seed: int,
) -> list[list[tuple[str, str]]]:
    random.seed(seed)
    torch.manual_seed(seed)

    if shuffle_auxiliary_function_name:
        function_names = df["function1_name"].tolist()
    if shuffle_auxiliary_function_docstring:
        docstrings = [
            parse_docstring(row["function1_human"])
            for _, row in df.iterrows()
        ]

    total_implementations = []
    for _, row in tqdm(df.iterrows(), total=df.shape[0],
                       desc="Implementation"):
        # Create stop tokens
        stop_tokens = row["stop_tokens"]

        # Create prompts
        prompts = []
        for _ in range(num_implementations):
            import_statements = row["imports"]
            auxiliary_function = row["function1_human"].strip()
            target_function_signature = row["function2_signature"].strip()
            if shuffle_auxiliary_function_name:
                original_name = row["function1_name"]
                replace_name = random.choice(function_names)
                while replace_name == original_name:
                    replace_name = random.choice(function_names)
                auxiliary_function = replace_function_name(
                    auxiliary_function, replace_name)
            if shuffle_auxiliary_function_docstring:
                original_docstring = parse_docstring(row["function1_human"])
                replace_docstring = random.choice(docstrings)
                while replace_docstring == original_docstring:
                    replace_docstring = random.choice(docstrings)
                auxiliary_function = replace_function_docstring(
                    auxiliary_function, replace_docstring)
            if remove_auxiliary_function_docstring:
                auxiliary_function = remove_function_docstring(
                    auxiliary_function)
            prompt = (f"{import_statements}\n\n\n"
                      f"{auxiliary_function}\n\n\n{target_function_signature}")
            prompts.append(prompt)

        # Generate implementation
        example_implementations = []
        for prompt_idx in range(0, len(prompts), 4):
            batch_prompts = prompts[prompt_idx:prompt_idx + 4]
            batch_completions = generate_fn(
                batch_prompts,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                stop_sequences=stop_tokens,
                num_return_sequences=4,
            )
            batch_completions = [(completion, finish_reason)
                                 for element in batch_completions
                                 for completion, finish_reason in element]

            # Parse completion to extract implementation
            for prompt, (text, finish_reason) in zip(batch_prompts,
                                                     batch_completions):
                implementation = remove_after_stop_token(
                    text, prompt, stop_tokens)
                if finish_reason != "length":
                    status = "ok"
                else:
                    status = "max generation token reach"
                example_implementations.append((implementation, [status]))

        total_implementations.append(example_implementations)

    return total_implementations
