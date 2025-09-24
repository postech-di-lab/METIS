import ast
import asyncio
import json
import os
import re
from typing import Callable
import jinja2
import fire
import pandas as pd
from openai import AsyncOpenAI

from datasets import load_dataset
from vllm import LLM, SamplingParams

BASE_MODELS = {
    # meta-llama
    "meta-llama/CodeLlama-7b-hf",
    "meta-llama/CodeLlama-34b-hf",
    "meta-llama/CodeLlama-70b-hf",
    "meta-llama/Meta-Llama-3-8B",
    "meta-llama/Meta-Llama-3-70B",
    # deepseek-ai
    "deepseek-ai/deepseek-coder-6.7b-base",
    "deepseek-ai/deepseek-coder-7b-base-v1.5",
    "deepseek-ai/deepseek-coder-33b-base",
    # google
    "google/codegemma-7b",
    # bigcode
    "bigcode/starcoder2-15b",
}

MODEL2TEMPLATE = {
    # meta-llama
    "meta-llama/CodeLlama-7b-hf": "code.j2",
    "meta-llama/CodeLlama-34b-hf": "code.j2",
    "meta-llama/CodeLlama-70b-hf": "code.j2",
    "meta-llama/Meta-Llama-3-8B": "code.j2",
    "meta-llama/Meta-Llama-3-70B": "code.j2",
    "meta-llama/CodeLlama-7b-Instruct-hf": "inst-codellama.j2",
    "meta-llama/CodeLlama-34b-Instruct-hf": "inst-codellama.j2",
    "meta-llama/CodeLlama-70b-Instruct-hf": "inst-codellama-70b.j2",
    "meta-llama/Meta-Llama-3-8B-Instruct": "inst-llama3.j2",
    "meta-llama/Meta-Llama-3-70B-Instruct": "inst-llama3.j2",
    # deepseek-ai
    "deepseek-ai/deepseek-coder-6.7b-base": "code.j2",
    "deepseek-ai/deepseek-coder-7b-base-v1.5": "code.j2",
    "deepseek-ai/deepseek-coder-33b-base": "code.j2",
    "deepseek-ai/deepseek-coder-6.7b-instruct": "inst-deepseekcoder.j2",
    "deepseek-ai/deepseek-coder-7b-instruct-v1.5": "inst-deepseekcoder.j2",
    "deepseek-ai/deepseek-coder-33b-instruct": "inst-deepseekcoder.j2",
    # google
    "google/codegemma-7b": "code.j2",
    "google/codegemma-7b-it": "inst-codegemma.j2",
    "google/codegemma-1.1-7b-it": "inst-codegemma.j2",
    # bigcode
    "bigcode/starcoder2-15b": "code.j2",
    "bigcode/starcoder2-15b-instruct-v0.1": "inst-starcoder2.j2",
    # ise-uiuc
    "ise-uiuc/Magicoder-S-CL-7B": "inst-majicoder.j2",
    "ise-uiuc/Magicoder-S-DS-6.7B": "inst-majicoder.j2",
    # Bin12345
    "Bin12345/AutoCoder_S_6.7B": "inst-autocoder.j2",
    "Bin12345/AutoCoder": "inst-autocoder.j2",
}

OPENAI_MODELS = {"gpt-3.5-turbo-0125", "gpt-4o-2024-05-13", "gpt-4-turbo-2024-04-09"}

jinja_env = jinja2.Environment(
    trim_blocks=True, lstrip_blocks=False, keep_trailing_newline=True,
    loader=jinja2.FileSystemLoader("templates"), autoescape=jinja2.select_autoescape())

def render_fn(template_path: str):
    template = jinja_env.get_template(template_path)

    def render(placeholder: dict) -> str:
        return template.render(**placeholder)

    return render


def parse_fn(template_path: str):
    if template_path == "inst-codellama.j2":
        pat = re.compile(r'\[INST\] (?P<context>.*?) \[\/INST\](?P<response>.*)', flags=re.M|re.S)
    elif template_path == "inst-majicoder.j2":
        pat = re.compile(r'You are an exceptionally intelligent coding assistant that consistently delivers accurate and reliable responses to user instructions\.\n\n@@ Instruction\n(?P<context>.*?)\n\n@@ Response\n(?P<response>.*)', flags=re.M|re.S)
    elif template_path == "inst-deepseekcoder.j2":
        pat = re.compile(r'You are an AI programming assistant, utilizing the Deepseek Coder model, developed by Deepseek Company, and you only answer questions related to computer science\. For politically sensitive questions, security and privacy issues, and other non-computer science questions, you will refuse to answer\n\n### Instruction:\n(?P<context>.*?)\n### Response:\n(?P<response>.*)', flags=re.M|re.S)
    elif template_path == "inst-autocoder.j2":
        pat = re.compile(r'Human: (?P<context>.*?)\nAssistant:(?P<response>.*)', flags=re.M|re.S)
    elif template_path == "inst-codegemma.j2":
        pat = re.compile(r'<start_of_turn>user\n(?P<context>.*?)\n\n<end_of_turn>\n<start_of_turn>model\n(?P<response>.*)', flags=re.M|re.S)
    elif template_path == "inst-llama3.j2":
        pat = re.compile(r'<\|start_header_id\|>user<\|end_header_id\|>\n\n(?P<context>.*?)<\|eot_id\|><\|start_header_id\|>assistant<\|end_header_id\|>\n\n(?P<response>.*)', flags=re.M|re.S)
    elif template_path == "inst-starcoder2.j2":
        pat = re.compile(r'You are an exceptionally intelligent coding assistant that consistently delivers accurate and reliable responses to user instructions.\n\n### Instruction\n(?P<context>.*?)\n\n### Response\n(?P<response>.*)', flags=re.M|re.S)
    elif template_path == "inst-codellama-70b.j2":
        pat = re.compile(r'Source: user\n\n(?P<context>.*?) <step> Source: assistant\nDestination: user\n\n(?P<response>.*)', flags=re.M|re.S)

    def parse(text: str) -> dict[str, str] | None:
        if (m := pat.match(text)) is None:
            return None
        return m.groupdict()
        
    return parse


def _generate_openai(model_name: str, temperature: float, top_p: float, num_return_sequences: int,
                     max_tokens: int):
    client = AsyncOpenAI()

    async def _generate_example(prompt: str) -> tuple[list[str], dict]:
        completion = await client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            n=num_return_sequences,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )
        return [choice.message.content for choice in completion.choices], completion.usage.to_dict()

    async def _worker(queue_in: asyncio.Queue, queue_out: asyncio.Queue):
        while True:
            idx, prompt = await queue_in.get()
            queue_out.put_nowait((idx, await _generate_example(prompt)))
            queue_in.task_done()
            print(f"Complete example {idx}")

    async def _generate(prompts: list[str]) -> list[tuple[list[str], dict]]:
        queue_in, queue_out = asyncio.Queue(), asyncio.Queue()
        for idx, prompt in enumerate(prompts):
            queue_in.put_nowait((idx, prompt))
        workers = [asyncio.create_task(_worker(queue_in, queue_out)) for i in range(10)]
        await queue_in.join()
        for worker in workers:
            worker.cancel()
        await asyncio.gather(*workers, return_exceptions=True)
        outputs = []
        while not queue_out.empty():
            outputs.append(queue_out.get_nowait())
        outputs = [x for _, x in sorted(outputs)]
        return outputs

    def generate(prompts: list[str]) -> list[tuple[list[str], dict]]:
        return asyncio.run(_generate(prompts))

    return generate


def _generate_vllm(model_name: str,
                   temperature: float,
                   top_p: float,
                   num_return_sequences: int,
                   max_tokens: int,
                   stop: list[str]):
    llm = LLM(model=model_name,
              max_model_len=4096,
              max_num_seqs=128,
              enforce_eager=True)
    sampling_params = SamplingParams(temperature=temperature,
                                     top_p=top_p,
                                     max_tokens=max_tokens,
                                     n=num_return_sequences,
                                     stop=stop)

    def generate(prompts: list[str]) -> list[list[dict]]:
        completions = llm.generate(prompts, sampling_params)
        return [
            [
                {
                    "prompt_text": completion.prompt,
                    "prompt_token_ids": completion.prompt_token_ids,
                    "completion_text": output.text,
                    "completion_token_ids": output.token_ids,
                    "finish_reason": output.finish_reason
                }
                for output in completion.outputs
            ]
            for completion in completions
        ]

    return generate


def generate_fn(model_name: str,
                temperature: float,
                top_p: float,
                num_return_sequences: int,
                max_tokens: int,
                stop: list[str]):
    if model_name in OPENAI_MODELS:
        return _generate_openai(model_name, temperature, top_p, num_return_sequences, max_tokens)
    else:
        return _generate_vllm(model_name, temperature, top_p, num_return_sequences, max_tokens, stop)


def generate_implementations_fn(
        generate: Callable,
        model_name: str,
        response_prefix_style: str,
        tag_style: str,
        add_imports_to_code: bool,
        add_target_docstring_to_code: bool,
        add_auxiliary_function_to_code: bool,
        add_auxiliary_information_to_context: bool,
        add_codeblock_to_response_prefix: bool) -> Callable[[pd.DataFrame], pd.DataFrame]:

    # ----------- rendering --------------
    _render_code = render_fn(template_path="code.j2")

    def _create_code(row: dict) -> str:
        placeholder = {
            'target_function': {
                'declaration': row['function2_declaration'],
                'implementation': row['function2_implementation']
            },
            'suffix': ''
        }
        if add_imports_to_code:
            placeholder['imports'] = row['imports']
        if add_target_docstring_to_code:
            placeholder['target_function']['docstring'] = row['function2_docstring']
        if add_auxiliary_function_to_code:
            placeholder['auxiliary_function'] = {
                'declaration': row['function1_declaration'],
                'docstring': row['function1_docstring'],
                'implementation': row['function1_implementation']
            }
        return _render_code(placeholder)

    if model_name in BASE_MODELS:
        _create_prompt = _create_code
    else:
        _render_context = render_fn(template_path="humanextension-context.j2")

        if add_auxiliary_information_to_context:
            _render_code_aux = render_fn(template_path="code-aux.j2")

        def _create_context(row: dict) -> str:
            if add_auxiliary_information_to_context:
                placeholder = {
                    'declaration': row['function1_declaration'],
                    'docstring': row['function1_docstring'],
                    'implementation': row['function1_implementation']
                }
                code_aux = _render_code_aux(placeholder)
            placeholder = {
                'declaration': row['function2_declaration'],
                'docstring': row['function2_docstring'],
                'add_auxiliary_information': add_auxiliary_information_to_context,
                'tag_style': tag_style
            }
            if add_auxiliary_information_to_context:
                placeholder['auxiliary'] = {
                    'declaration': row['function1_declaration'],
                    'docstring': row['function1_docstring'],
                    'code': code_aux,
                }
            return _render_context(placeholder)

        if model_name in OPENAI_MODELS:
            # for openai models, provide a context
            _create_prompt = _create_context
        else:
            # for open-sourced models, provide a chat-format prompt
            _render_response_prefix = render_fn(template_path="humanextension-response-prefix.j2")

            if response_prefix_style == "default":
                response_prefix = ''
            elif response_prefix_style == "newline":
                response_prefix = '\n'
            elif response_prefix_style == "whitespace+newline":
                response_prefix = ' \n'
            else:
                raise ValueError()

            def _create_response_prefix(row: dict) -> str:
                code = _create_code(row)
                placeholder = {'code': code, 'prefix': response_prefix, 'add_codeblock': add_codeblock_to_response_prefix, 'tag_style': tag_style}
                return _render_response_prefix(placeholder)

            chat_template_path = MODEL2TEMPLATE[model_name]
            _render_chat = render_fn(template_path=chat_template_path)

            def _create_chat(row: dict) -> str:
                context = _create_context(row)
                response_prefix = _create_response_prefix(row)
                placeholder = {'context': context, 'response_prefix': response_prefix}
                return _render_chat(placeholder)

            _create_prompt = _create_chat

    # -------------- parsing ---------------
    def _extract_function(code: str, function_name: str) -> str | None:
        try:
            node = ast.parse(code)
        except:
            return None
        for n in node.body:
            if isinstance(n, ast.FunctionDef) and n.name == function_name:
                return ast.unparse(n)
        return None

    if model_name in BASE_MODELS:
        def _parse_target_function_strict(text: str, function_name: str) -> str | None:
            return _extract_function(text, function_name)
        def _parse_target_function_flexible(text: str, function_name: str) -> str | None:
            return _extract_function(text, function_name)
    else:
        if tag_style == "markdown":
            codeblock_tag = re.compile(r'.*?(```python|```)\n(?P<code>.*?)```', flags=re.M|re.S)
        else:
            codeblock_tag = re.compile(r'.*?\[PYTHON\]\n(?P<code>.*?)\[\/PYTHON\]', flags=re.M|re.S)
        codeblock_tag_resilient = re.compile(r'.*?(\[PYTHON\]\n```python|\[PYTHON\]\n```|\[PYTHON\]|```python|```)\n(?P<code>.*?)(\[\/PYTHON\]\n```|\[\/PYTHON\]|```)', flags=re.M|re.S)
        if model_name in OPENAI_MODELS:
            # openai models provide response
            def _parse_target_function_strict(text: str, function_name: str) -> str | None:
                if (m := codeblock_tag.match(text)) is None:
                    return None
                return _extract_function(m.group('code'), function_name)
            def _parse_target_function_flexible(text: str, function_name: str) -> str | None:
                if (m := codeblock_tag_resilient.match(text)) is None:
                    return None
                return _extract_function(m.group('code'), function_name)
        else:
            chat_template_path = MODEL2TEMPLATE[model_name]
            _parse_chat = parse_fn(template_path=chat_template_path)
            def _parse_target_function_strict(text: str, function_name: str) -> str | None:
                if (m := _parse_chat(text)) is None:
                    return None
                if (m := codeblock_tag.match(m['response'])) is None:
                    return None
                return _extract_function(m.group('code'), function_name)
            def _parse_target_function_flexible(text: str, function_name: str) -> str | None:
                if (m := _parse_chat(text)) is None:
                    return None
                if (m := codeblock_tag_resilient.match(m['response'])) is None:
                    return None
                return _extract_function(m.group('code'), function_name)

    def _parse(row: dict) -> dict:
        code_strict = _parse_target_function_strict(row['prompt'] + row['completion_text'],
                                                    row['function2_name'])
        code_flexible = _parse_target_function_flexible(row['prompt'] + row['completion_text'],
                                                        row['function2_name'])
        return {'code_strict': code_strict, 'code_flexible': code_flexible}

    def generate_implementations(df: pd.DataFrame) -> pd.DataFrame:
        df['prompt'] = df.apply(_create_prompt, axis=1)
        print("--prompt--")
        print(df.loc[0, 'prompt'])
        print("--end--")
        df['completion'] = generate(df['prompt'].tolist())
        df = df.explode('completion').reset_index(names=['problem_number'])
        df = pd.concat((df.drop('completion', axis=1), pd.json_normalize(df['completion'])), axis=1)
        df = pd.concat((df, df.apply(_parse, axis=1, result_type='expand')), axis=1)
        return df

    return generate_implementations


def main(model_name: str,
         save_path: str,
         response_prefix_style: str = "default",
         tag_style: str = "default",
         add_imports_to_code: bool = False,
         add_target_docstring_to_code: bool = True,
         add_auxiliary_function_to_code: bool = False,
         add_auxiliary_information_to_context: bool = False,
         add_codeblock_to_response_prefix: bool = False,
         temperature: float = 0.2,
         top_p: float = 0.95,
         max_tokens: int = 512,
         num_return_sequences: int = 20):
    
    print("-- Argument --")
    print(f"{model_name = }")
    print(f"{save_path = }")
    print(f"{response_prefix_style = }")
    print(f"{tag_style = }")
    print(f"{add_imports_to_code = }")
    print(f"{add_target_docstring_to_code = }")
    print(f"Approach 1. {add_auxiliary_information_to_context = }")
    print(f"Approach 2-1. {add_codeblock_to_response_prefix = }")
    print(f"Approach 2-2. {add_auxiliary_function_to_code = }")
    print("-- End --")

    stop = ["\ndef", "\nclass", "\nif", "\n#"] if model_name in BASE_MODELS else None

    generate = generate_fn(model_name=model_name,
                           temperature=temperature,
                           top_p=top_p,
                           num_return_sequences=num_return_sequences,
                           max_tokens=max_tokens,
                           stop=stop)

    generate_implementations = generate_implementations_fn(
        generate=generate, model_name=model_name,
        response_prefix_style=response_prefix_style,
        tag_style=tag_style,
        add_imports_to_code=add_imports_to_code,
        add_target_docstring_to_code=add_target_docstring_to_code,
        add_auxiliary_function_to_code=add_auxiliary_function_to_code,
        add_auxiliary_information_to_context=add_auxiliary_information_to_context,
        add_codeblock_to_response_prefix=add_codeblock_to_response_prefix)

    # load dataset
    df = load_dataset("sh0416/humanextension", split="test").to_pandas()

    # generate
    df_out = generate_implementations(df)

    # extract generated text
    configurations = {
        "model": model_name,
        "save_path": save_path,
        "response_prefix_style": response_prefix_style,
        "tag_style": tag_style,
        "add_imports_to_code": add_imports_to_code,
        "add_target_docstring_to_code": add_target_docstring_to_code,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
        "num_return_sequences": num_return_sequences,
    }

    # Save results
    os.makedirs(save_path, exist_ok=True)
    with open(os.path.join(save_path, 'configuration.json'), "w") as f:
        json.dump(configurations, f)
    df_out.to_parquet(os.path.join(save_path, 'results.parquet'))


if __name__ == "__main__":
    fire.Fire(main)
