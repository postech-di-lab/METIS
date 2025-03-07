import ast
import random
from functools import partial

import pandas as pd
import torch
from loguru import logger
from tqdm import tqdm


def parse_signature(code: str) -> str:
    node = ast.parse(code)
    for segment in node.body:
        if isinstance(segment, ast.FunctionDef):
            # Remove docstring
            segment.body = []
            # Remove "def "
            signature = ast.unparse(segment)[4:]
            return signature
    raise ValueError("Function signature not found")


def parse_docstring(code: dict) -> str:
    node = ast.parse(code)
    for segment in node.body:
        if isinstance(segment, ast.FunctionDef):
            docstring = ast.get_docstring(segment)
            return docstring
    raise ValueError("Function signature not found")


def create_prompts_step1(row: dict, num_implementations: int) -> list[str]:
    code = row["function1_signature"]
    signature = parse_signature(code).strip()
    docstring = parse_docstring(code).strip()
    import_statements = row["imports"].strip()
    target_function_signature = row["function1_signature"].strip()
    if len(import_statements) == 0:
        code = target_function_signature
    else:
        code = f"{import_statements}\n\n\n{target_function_signature}"
    prompt = f"<s>[INST] Write a Python function `{signature}` to solve the following problem:\n{docstring}\nYour code should start with a [PYTHON] tag and end with a [/PYTHON] tag.\n[/INST]\n[PYTHON]\n{code}"
    return [prompt for _ in range(num_implementations)]


def create_prompts_step2(completions: list[str], row: dict) -> list[str]:
    target_function_code = row["function2_signature"]
    target_function_signature = parse_signature(target_function_code)
    target_function_docstring = parse_docstring(target_function_code)
    prompts = []
    for completion, _ in completions:
        prompt = f"<s>{completion}</s><s>[INST] Write a Python function `{target_function_signature}` to solve the following problem:\n{target_function_docstring}\nYour code should start with a [PYTHON] tag and end with a [/PYTHON] tag.\nYou can use the above function whenever you needed.\n[/INST]\n[PYTHON]\n{target_function_code}"
        prompts.append(prompt)
    return prompts


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
        raise ValueError(f"Completion doesn't start with [INST]: {text}")
    implementation = ""
    index_start = 0
    while True:
        try:
            # Find next instruction
            index_end = text.index("[INST]", index_start + 1)
            turn = text[index_start:index_end]
            implementation_chunk, status = extract_implementation_from_turn(
                turn)
            implementation += f"{implementation_chunk}\n"
            index_start = index_end
            if status != "ok":
                return implementation, status
        except ValueError:
            turn = text[index_start:]
            implementation_chunk, status = extract_implementation_from_turn(
                turn)
            implementation += f"{implementation_chunk}\n"
            return implementation, status


def implement_instruct_step_by_step_humanextension(
    df: pd.DataFrame,
    generate_fn: callable,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    num_implementations: int,
    seed: int,
) -> list[list[tuple[str, str]]]:
    torch.manual_seed(seed)

    # Create prompt
    stop_tokens = ["[/PYTHON]"]
    total_implementations = []
    for _, row in tqdm(df.iterrows(), total=df.shape[0],
                       desc="Implementation"):
        # Generate auxiliary function
        prompts = create_prompts_step1(row, num_implementations)

        completions = []
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
            completions.extend(batch_completions)

        # Generate target function
        prompts = create_prompts_step2(completions, row)

        completions = []
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
            completions.extend(batch_completions)

        # Extract implementation
        implementations = []
        for text, _ in completions:
            implementation, status = extract_implementation(text)
            implementations.append((implementation, status))

        total_implementations.append(implementations)
    return total_implementations
