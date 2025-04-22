import random

import pandas as pd
import torch
from loguru import logger
from tqdm import tqdm

from humanextension.utils import create_humaneval_functions, remove_after_stop_token


def implement_irrelevant_humanextension(
    df: pd.DataFrame,
    generate_fn: callable,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    num_implementations: int,
    seed: int,
) -> list[list[tuple[str, str]]]:
    torch.manual_seed(seed)
    # Create auxiliary function pool
    humaneval_functions = create_humaneval_functions()

    total_implementations = []
    for _, row in tqdm(df.iterrows(), total=df.shape[0],
                       desc="Implementation"):
        # Sample random auxiliary functions
        auxiliary_functions = random.sample(humaneval_functions,
                                            k=num_implementations)

        # Create prompts
        prompts = []
        for auxiliary_import_statements, auxiliary_function_definition in auxiliary_functions:
            import_statements = set(row["imports"].split("\n"))
            import_statements.update(auxiliary_import_statements)
            import_statements = "\n".join(sorted(import_statements))

            prompts.append(f"{import_statements}\n\n\n"
                           f"{auxiliary_function_definition.strip()}\n\n\n"
                           f"{row['function2_signature'].strip()}")

        stop_tokens = row["stop_tokens"]

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
                num_return_sequences=1,
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
