import os
from typing import Callable

import torch
import requests
from loguru import logger
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    StoppingCriteria,
    StoppingCriteriaList,
)


TORCH_DTYPES = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


class EndOfFunctionCriteria(StoppingCriteria):
    """Custom `StoppingCriteria` which checks if all generated functions in the batch are completed."""

    def __init__(self, start_length, eof_strings, tokenizer):
        self.start_length = start_length
        self.eof_strings = eof_strings
        self.tokenizer = tokenizer
        self.check_fn = lambda decoded_generation: any([
            stop_string in decoded_generation
            for stop_string in self.eof_strings
        ])

    def __call__(self, input_ids, scores, **kwargs):
        """Returns true if all generated sequences contain any of the end-of-function strings."""
        decoded_generations = self.tokenizer.batch_decode(
            input_ids[:, self.start_length:])
        return all([
            self.check_fn(decoded_generation)
            for decoded_generation in decoded_generations
        ])


def create_tokenizer(checkpoint_name: str):
    params = {
        "pretrained_model_name_or_path": checkpoint_name,
        "trust_remote_code": True,
    }
    if checkpoint_name.startswith("facebook/incoder"):
        params["pad_token"] = "<pad>"
        params["unk_token"] = "<unk>"
    elif checkpoint_name.startswith("Salesforce/codegen"):
        params["pad_token"] = "<|endoftext|>"  # eos token
    elif checkpoint_name == "bigcode/santacoder":
        params["pad_token"] = "<|endoftext|>"
    return AutoTokenizer.from_pretrained(**params)


def create_model(checkpoint_name: str, dtype: str):
    params = {
        "pretrained_model_name_or_path": checkpoint_name,
        "torch_dtype": TORCH_DTYPES[dtype],
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
    }
    if checkpoint_name == "facebook/incoder-6B" and dtype == "fp16":
        params["revision"] = "float16"
    if checkpoint_name == "Salesforce/codegen-16B-multi":
        params["revision"] = "sharded"
    if checkpoint_name == "Salesforce/codegen-16B-mono":
        params["revision"] = "sharded"
    if checkpoint_name.startswith("codellama/CodeLlama"):
        params["use_flash_attention_2"] = True
    return AutoModelForCausalLM.from_pretrained(**params)


def create_generate_hf(checkpoint_name: str, dtype: str) -> Callable:
    logger.info(f"Checkpoint name: {checkpoint_name}, dtype: {dtype}")

    # Load tokenizer and model
    tokenizer = create_tokenizer(checkpoint_name)

    # Get pad token for batch generation
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"

    # NOTE: We use eos token as pad token if pad token is not available
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer must have a pad token or an eos token")
        tokenizer.pad_token = tokenizer.eos_token

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model(checkpoint_name, dtype).eval()
    model = model.to(device)

    def generate(
        prompts: list[str],
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        stop_sequences: list[str],
        num_return_sequences: int,
    ) -> list[list[tuple[str, str]]]:
        if temperature == 0.0:
            generation_config = GenerationConfig(
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id,
                use_cache=True,
                num_return_sequences=num_return_sequences,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
        else:
            generation_config = GenerationConfig(
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id,
                use_cache=True,
                num_return_sequences=num_return_sequences,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_k=None,
                top_p=top_p,
            )
        with torch.inference_mode():
            encodings = tokenizer(
                prompts,
                padding=True,
                truncation=True,
                return_token_type_ids=False,
                return_tensors="pt",
            )
            encodings = encodings.to(device)
            stopping_criteria = StoppingCriteriaList([
                EndOfFunctionCriteria(
                    start_length=encodings["input_ids"].shape[1],
                    eof_strings=stop_sequences,
                    tokenizer=tokenizer,
                )
            ])
            output_ids = model.generate(
                **encodings,
                generation_config=generation_config,
                stopping_criteria=stopping_criteria,
            )
            output_ids = output_ids.tolist()
            completions = tokenizer.batch_decode(
                output_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )

        def determine_finish_reason(text: str, output_id: list[int]):
            if any(
                    text.endswith(stop_sequence)
                    for stop_sequence in stop_sequences):
                return "stop_sequence"
            elif tokenizer.eos_token_id in output_id:
                return "eos_token"
            else:
                return "length"

        completions = [(text, determine_finish_reason(text, output_id))
                       for text, output_id in zip(completions, output_ids)]
        completions = [
            completions[idx:idx + num_return_sequences]
            for idx in range(0,
                             len(prompts) *
                             num_return_sequences, num_return_sequences)
        ]
        return completions

    return generate


def create_generate_api_completion(api_url: str, model_name: str, openai: bool=False) -> Callable:
    logger.info(f"API url: {api_url}, model name: {model_name}")

    headers = {"Content-Type": "application/json"}
    if openai:
        headers["Authorization"] = f"Bearer {os.environ.get(['OPENAI_API_KEY'])}"
        headers['OpenAI-Organization'] = os.environ.get(['OPENAI_ORG_ID'])

    def generate(
        prompts: list[str],
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        stop_sequences: list[str],
        num_return_sequences: int,
    ) -> list[list[tuple[str, str]]]:
        payload = {
            "model": model_name,
            "prompt": prompts,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_new_tokens,
            "stop": stop_sequences,
            "n": num_return_sequences,
        }
        response = requests.post(api_url, headers=headers, json=payload)
        completions = [(choice['text'], choice['finish_reason']) for choice in response.json()['choices']]

        completions = [
            completions[idx:idx + num_return_sequences]
            for idx in range(0, len(prompts) * num_return_sequences, num_return_sequences)
        ]
        return completions

    return generate


def create_generate_api_chat(api_url: str, model_name: str, openai: bool=False) -> Callable:
    logger.info(f"API url: {api_url}, model name: {model_name}")

    headers = {"Content-Type": "application/json"}
    if openai:
        headers["Authorization"] = f"Bearer {os.environ.get(['OPENAI_API_KEY'])}"
        headers['OpenAI-Organization'] = os.environ.get(['OPENAI_ORG_ID'])

    def generate(
        messages_list: list[list[dict]],
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        stop_sequences: list[str],
        num_return_sequences: int,
    ) -> list[list[tuple[str, str]]]:
        completions = []
        for messages in messages_list:
            payload = {
                "model": model_name,
                "messages": messages,
                "temperature": temperature,
                "top_p": top_p,
                "max_tokens": max_new_tokens,
                "stop": stop_sequences,
                "n": num_return_sequences,
            }
            response = requests.post(api_url, headers=headers, json=payload)
            completions.append([(choice['message']['content'], choice['finish_reason']) for choice in response.json()['choices']])
        return completions

    return generate
