import random

import pandas as pd
import torch
from loguru import logger
from tqdm import tqdm

from humanextension.utils import create_humaneval_functions, remove_after_stop_token


def implement_multiple_auxiliary_functions_humanextension(
    df: pd.DataFrame,
    generate_fn: callable,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    num_implementations: int,
    num_auxiliary_functions: int,
    relevant_auxiliary_function_position: int,
    seed: int,
) -> list[list[tuple[str, str]]]:
    random.seed(seed)
    torch.manual_seed(seed)

    # Create auxiliary function pool
    humaneval_functions = create_humaneval_functions()

    total_implementations = []
    for _, row in tqdm(df.iterrows(), total=df.shape[0],
                       desc="Implementation"):
        # Create stop tokens
        stop_tokens = row["stop_tokens"]

        # Create prompts
        prompts = []
        for _ in range(num_implementations):
            # Sample k auxiliary functions
            distractor_functions = random.sample(humaneval_functions,
                                                 k=num_auxiliary_functions - 1)

            # Relevant auxiliary function
            relevant_function = row["function1_human"]

            # Create import statements
            import_statements = set(row["imports"].split("\n"))
            for distractor_import_statements, _ in distractor_functions:
                import_statements.update(distractor_import_statements)
            import_statements = "\n".join(sorted(import_statements))

            auxiliary_functions = [
                implementation for _, implementation in distractor_functions
            ]
            auxiliary_functions.insert(relevant_auxiliary_function_position,
                                       relevant_function)
            auxiliary_functions = "\n\n\n".join(f.strip()
                                                for f in auxiliary_functions)

            prompt = (f"{import_statements}\n\n\n"
                      f"{auxiliary_functions}\n\n\n"
                      f"{row['function2_signature'].strip()}")
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
