import ast
import random
from functools import partial

import pandas as pd
import torch
from loguru import logger
from tqdm import tqdm

from humanextension.utils import create_humaneval_functions, parse_signature, parse_docstring


def create_prompt(row: dict) -> str:
    prompt = row["prompt"]
    signature = parse_signature(row["prompt"])
    docstring = parse_docstring(row["prompt"])
    prompt = f"<s> [INST] Write a Python function `{signature}` to solve the following problem:\n{docstring}\nYour code should start with a [PYTHON] tag and end with a [/PYTHON] tag.\n[/INST]\n[PYTHON]\n{prompt}"
    return prompt


def create_prompts_humanextension_direct(row: dict, num_implementations: int) -> list[str]:
    code = row["function2_signature"]
    signature = parse_signature(code)
    docstring = parse_docstring(code)
    import_statements = row["imports"].strip()
    target_function_signature = row["function2_signature"].strip()
    if len(import_statements) == 0:
        code = target_function_signature
    else:
        code = f"{import_statements}\n\n\n{target_function_signature}"
    prompt = f"<s> [INST] Write a Python function `{signature}` to solve the following problem:\n{docstring}\nYour code should start with a [PYTHON] tag and end with a [/PYTHON] tag.\n[/INST]\n[PYTHON]\n{code}"
    return [prompt for _ in range(num_implementations)]


def create_prompts_humanextension_irrelevant(row: dict, humaneval_functions: list[tuple[tuple[str, ...], str]], num_implementations: int) -> list[str]:
    target_function_signature = parse_signature(row["function2_signature"])
    target_function_docstring = parse_docstring(row["function2_signature"])
    target_function_code = row["function2_signature"]

    # Sample random auxiliary functions
    auxiliary_functions = random.sample(humaneval_functions, k=num_implementations)

    # Create prompts
    prompts = []
    for auxiliary_import_statements, auxiliary_function_definition in auxiliary_functions:
        auxiliary_function_signature = parse_signature(auxiliary_function_definition)
        auxiliary_function_docstring = parse_docstring(auxiliary_function_definition)

        # Create code for auxiliary function
        import_statements = set(row["imports"].split("\n"))
        import_statements.update(auxiliary_import_statements)
        import_statements = "\n".join(sorted(import_statements))
        auxiliary_function_code = f"{import_statements}\n\n\n{auxiliary_function_definition.strip()}"

        prompt = f"<s>[INST] Write a Python function `{auxiliary_function_signature}` to solve the following problem:\n{auxiliary_function_docstring}\nYour code should start with a [PYTHON] tag and end with a [/PYTHON] tag.\n[/INST]\n[PYTHON]\n{auxiliary_function_code}\n[/PYTHON]\n</s><s>[INST] Write a Python function `{target_function_signature}` to solve the following problem:\n{target_function_docstring}\nYour code should start with a [PYTHON] tag and end with a [/PYTHON] tag.\n[/INST]\n[PYTHON]\n{target_function_code}"

        prompts.append(prompt)
    return prompts


def create_prompts_humanextension_oracle(row: dict, num_implementations: int) -> list[str]:
    # Create prompt
    import_statements = row["imports"]
    auxiliary_function_code = row["function1_human"]
    auxiliary_function_signature = parse_signature(auxiliary_function_code)
    auxiliary_function_docstring = parse_docstring(auxiliary_function_code)
    target_function_code = row["function2_signature"]
    target_function_signature = parse_signature(target_function_code)
    target_function_docstring = parse_docstring(target_function_code)
    prompt = f"<s>[INST] Write a Python function `{auxiliary_function_signature}` to solve the following problem:\n{auxiliary_function_docstring}\nYour code should start with a [PYTHON] tag and end with a [/PYTHON] tag.\n[/INST]\n[PYTHON]\n{import_statements}\n\n\n{auxiliary_function_code}\n[/PYTHON]\n</s><s>[INST] Write a Python function `{target_function_signature}` to solve the following problem:\n{target_function_docstring}\nYour code should start with a [PYTHON] tag and end with a [/PYTHON] tag.\nYou can use the above function whenever you needed.\n[/INST]\n[PYTHON]\n{target_function_code}"
    return [prompt for _ in range(num_implementations)]


def extract_implementation_from_turn(text: str) -> tuple[str, str]:
    try:
        index = text.index("[/INST]") + len("[/INST]")
        index_start = text.index("[PYTHON]", index) + len("[PYTHON]")
    except ValueError:
        raise ValueError("Given format is incorrect. Check prompt.")

    try:
        index_end = text.index("[/PYTHON]", index)
        text = text[index_start:index_end]
        return text, "ok"
    except ValueError:
        text = text[index_start:]
        return text, "[/PYTHON] tag not found"


def extract_implementation(text: str) -> tuple[str, str]:
    text = text.strip()
    if not text.startswith("[INST]"):
        raise ValueError("Completion doesn't start with [INST]")
    implementation = ""
    index_start = 0
    while True:
        try:
            # Find next instruction
            index_end = text.index("[INST]", index_start + 1)
            turn = text[index_start:index_end]
            implementation_chunk, status = extract_implementation_from_turn(turn)
            implementation += f"{implementation_chunk}\n"
            index_start = index_end
            if status != "ok":
                return implementation, status
        except ValueError:
            turn = text[index_start:]
            implementation_chunk, status = extract_implementation_from_turn(turn)
            implementation += f"{implementation_chunk}\n"
            return implementation, status


def implement_instruct_humaneval(
    df: pd.DataFrame,
    generate_fn: callable,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
    num_implementations: int,
    seed: int,
) -> list[list[tuple[str, str]]]:
    torch.manual_seed(seed)

    # Create prompt
    prompts = df.apply(create_prompt, axis=1)
    stop_tokens = ["[/PYTHON]"]

    total_implementations = []
    for prompt in tqdm(prompts, desc="Implementation"):
        example_implementations = []
        while len(example_implementations) < num_implementations:
            # Generate implementation
            completions = generate_fn(
                [prompt],
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                stop_sequences=stop_tokens,
                num_return_sequences=4,
            )[0]
            # Extract implementation
            for text, _ in completions:
                implementation, status = extract_implementation(text)
                example_implementations.append((implementation, status))
        total_implementations.append(example_implementations)
    return total_implementations


def implement_instruct_humanextension(
    df: pd.DataFrame,
    generate_fn: callable,
    method_name: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    num_implementations: int,
    seed: int,
) -> list[list[tuple[str, str]]]:
    torch.manual_seed(seed)

    # Create prompt
    if method_name == "instruct-direct":
        create_prompts = partial(
            create_prompts_humanextension_direct,
            num_implementations=num_implementations,
        )
    elif method_name == "instruct-irrelevant":
        humaneval_functions = create_humaneval_functions()
        create_prompts = partial(
            create_prompts_humanextension_irrelevant,
            humaneval_functions=humaneval_functions,
            num_implementations=num_implementations,
        )
    elif method_name == "instruct-oracle":
        create_prompts = partial(
            create_prompts_humanextension_oracle,
            num_implementations=num_implementations,
        )
    else:
        raise ValueError(f"Unknown method: {method_name}")

    stop_tokens = ["[/PYTHON]"]
    total_implementations = []
    for _, row in tqdm(df.iterrows(), total=df.shape[0],
                       desc="Implementation"):
        # Create prompts
        prompts = create_prompts(row)

        example_implementations = []
        for i in range(0, len(prompts), 4):
            batch_prompts = prompts[i:i + 4]
            # Generate implementation
            batch_completions = generate_fn(
                batch_prompts,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                stop_sequences=stop_tokens,
                num_return_sequences=1,
            )
            batch_completions = [y for x in batch_completions for y in x]

            # Extract implementation
            for text, _ in batch_completions:
                implementation, status = extract_implementation(text)
                example_implementations.append((implementation, status))
        total_implementations.append(example_implementations)
    return total_implementations
