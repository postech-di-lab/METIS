import argparse
import json
import os
import sys
from datetime import datetime

import torch
from loguru import logger

from humanextension.dataset import load_humaneval_multiple, load_humanextension
from humanextension.evaluation import evaluate_functional_correctness
from humanextension.generation import create_generate_hf, create_generate_api
from humanextension.pipeline import (
    implement_direct_humaneval,
    implement_direct_humanextension,
    implement_instruct_humaneval,
    implement_instruct_humanextension,
    implement_instruct_step_by_step_humanextension,
    implement_irrelevant_humanextension,
    implement_oracle_humanextension,
    implement_step_by_step_humanextension,
    implement_multiple_auxiliary_functions_humanextension,
)
from humanextension.utils import estimate_pass_at_k

# fmt: off
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, required=True, choices=["humaneval", "humanextension"])
parser.add_argument("--method", type=str, choices=["direct", "irrelevant", "step_by_step", "oracle", "instruct-direct", "instruct-irrelevant", "instruct-oracle", "instruct-step_by_step", "multiple_auxiliary_functions"])
parser.add_argument("--model", type=str, required=True)
parser.add_argument("--model_dtype", type=str, default='float16', choices=['float16', 'bfloat16', 'float32'])
parser.add_argument("--model_api_url", type=str)
parser.add_argument("--max_new_tokens", type=int, default=512)
parser.add_argument("--temperature", type=float, default=0.2)
parser.add_argument("--top_p", type=float, default=0.95)
parser.add_argument("--num_completions", type=int, default=10)
parser.add_argument("--num_return_sequences", type=int, default=4)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--num_auxiliary_functions", type=int, default=1, help="Use for multiple auxiliary functions setting")
parser.add_argument("--relevant_auxiliary_function_position", type=int, default=0, help="Use for multiple auxiliary functions setting")
parser.add_argument("--shuffle_auxiliary_function_name", action="store_true", help="Whether to replace auxiliary function name to random. Only work when oracle setting.")
parser.add_argument("--remove_auxiliary_function_docstring", action="store_true", help="Whetehr to remove auxiliary function docstring. Only work when oracle setting.")
parser.add_argument("--output_dirpath", type=str, required=True)
# fmt: on


def main():
    logger.remove()
    logger.add(sys.stderr, level="INFO")

    logger.info(f"Cuda availability: {torch.cuda.is_available()}")

    args = parser.parse_args()
    logger.info(f"Configuration: {args}")

    output_dirpath = os.path.join(args.output_dirpath, args.dataset,
                                  args.model,
                                  datetime.now().isoformat())
    logger.info(f"output path (considering args.ration): {output_dirpath}")

    logger.info("Create output path if it doesn't exist.")
    os.makedirs(output_dirpath, exist_ok=True)

    logger.info("Save configuration.")
    with open(os.path.join(output_dirpath, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    logger.info("Load dataset")
    if args.dataset == "humaneval":
        df = load_humaneval_multiple()
    elif args.dataset == "humanextension":
        df = load_humanextension()
    else:
        raise NotImplementedError()

    logger.info("create completion function")
    if args.model_api_url:
        generate_fn = create_generate_api(args.model_api_url, args.model)
    else:
        generate_fn = create_generate_hf(args.model, args.model_dtype)

    if args.method == "direct":
        if args.dataset == "humaneval":
            implement_fn = implement_direct_humaneval
        elif args.dataset == "humanextension":
            implement_fn = implement_direct_humanextension
        else:
            raise ValueError()
        outputs = implement_fn(df, generate_fn=generate_fn, temperature=args.temperature, top_p=args.top_p, max_new_tokens=args.max_new_tokens, num_implementations=args.num_completions, num_return_sequences=args.num_return_sequences, seed=args.seed)
    elif args.method == "instruct" and args.dataset == "humaneval":
        outputs = implement_instruct_humaneval(
            df,
            generate_fn,
            temperature=args.temperature,
            top_p=args.top_p,
            max_new_tokens=args.max_new_tokens,
            num_implementations=args.num_completions,
            seed=args.seed)
    elif args.method == "instruct-step_by_step" and args.dataset == "humanextension":
        outputs = implement_instruct_step_by_step_humanextension(
            df,
            generate_fn,
            temperature=args.temperature,
            top_p=args.top_p,
            max_new_tokens=args.max_new_tokens,
            num_implementations=args.num_completions,
            seed=args.seed)
    elif args.method.startswith(
            "instruct") and args.dataset == "humanextension":
        outputs = implement_instruct_humanextension(
            df,
            generate_fn,
            method_name=args.method,
            temperature=args.temperature,
            top_p=args.top_p,
            max_new_tokens=args.max_new_tokens,
            num_implementations=args.num_completions,
            seed=args.seed)
    elif args.method == "irrelevant" and args.dataset == "humanextension":
        outputs = implement_irrelevant_humanextension(
            df,
            generate_fn,
            temperature=args.temperature,
            top_p=args.top_p,
            max_new_tokens=args.max_new_tokens,
            num_implementations=args.num_completions,
            seed=args.seed)
    elif args.method == "step_by_step" and args.dataset == "humanextension":
        outputs = implement_step_by_step_humanextension(
            df,
            generate_fn,
            temperature=args.temperature,
            top_p=args.top_p,
            max_new_tokens=args.max_new_tokens,
            num_implementations=args.num_completions,
            seed=args.seed)
    elif args.method == "oracle" and args.dataset == "humanextension":
        outputs = implement_oracle_humanextension(
            df,
            generate_fn,
            temperature=args.temperature,
            top_p=args.top_p,
            max_new_tokens=args.max_new_tokens,
            num_implementations=args.num_completions,
            shuffle_auxiliary_function_name=args.
            shuffle_auxiliary_function_name,
            shuffle_auxiliary_function_docstring=args.
            shuffle_auxiliary_function_docstring,
            remove_auxiliary_function_docstring=args.
            remove_auxiliary_function_docstring,
            seed=args.seed)
    elif args.method == "multiple_auxiliary_functions" and args.dataset == "humanextension":
        outputs = implement_multiple_auxiliary_functions_humanextension(
            df,
            generate_fn,
            temperature=args.temperature,
            top_p=args.top_p,
            max_new_tokens=args.max_new_tokens,
            num_implementations=args.num_completions,
            num_auxiliary_functions=args.num_auxiliary_functions,
            relevant_auxiliary_function_position=args.
            relevant_auxiliary_function_position,
            seed=args.seed)
    else:
        raise NotImplementedError()
    df["implementations"] = [[code for code, _ in example] for example in outputs]
    df["status"] = [[status for _, status in example] for example in outputs]

    logger.info("Evaluate functional correctness")
    df["functional_correctness"] = df.apply(evaluate_functional_correctness, axis=1)

    def compute_num_correct(row: dict):
        num_correct = 0
        for functional_correctness in row["functional_correctness"]:
            if functional_correctness == "passed":
                num_correct += 1
        return num_correct

    logger.info("Compute metric values")
    df["num_correct"] = df.apply(compute_num_correct, axis=1)
    df["num_candidate"] = df["functional_correctness"].map(lambda l: len(l))

    df[f"pass@1"] = df.apply(lambda x: estimate_pass_at_k(x["num_candidate"], x["num_correct"], 1), axis=1)
    df[f"pass@10"] = df.apply(lambda x: estimate_pass_at_k(x["num_candidate"], x["num_correct"], 10), axis=1)

    # 결과 저장
    with open(os.path.join(output_dirpath, "results_example.jsonl"), "w") as f:
        df.to_json(f, orient="records", lines=True)

    df_aggregate = df.mean(numeric_only=True)
    with open(os.path.join(output_dirpath, "results_aggregated.json"), "w") as f:
        df_aggregate.to_json(f, orient="index")

    logger.info("Finish")


if __name__ == "__main__":
    main()
