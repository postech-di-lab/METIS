# HumanExtension

A evaluation dataset for measuring code generation capability with **auxiliary function**. There exist two research work that evaluate code language model and their intstruction-tuned variants.

* Seonghyeon Lee, Sanghwan Jang, Seongbo Jang, Dongha Lee, and Hwanjo Yu. 2024. [Exploring Language Model’s Code Generation Ability with Auxiliary Functions](https://aclanthology.org/2024.findings-naacl.181/). In Findings of the Association for Computational Linguistics: NAACL 2024, pages 2836–2848, Mexico City, Mexico. Association for Computational Linguistics.
* Seonghyeon Lee, Suyeon Kim, Joonwon Jang, HeeJae Chon, Dongha Lee, and Hwanjo Yu. 2024. [Eliciting Instruction-tuned Code Language Models’ Capabilities to Utilize Auxiliary Function for Code Generation](https://aclanthology.org/2024.findings-emnlp.100/). In Findings of the Association for Computational Linguistics: EMNLP 2024, pages 1840–1846, Miami, Florida, USA. Association for Computational Linguistics.

### What is auxiliary function?

Auxiliary function is a function that helps implement other function that is of our interest. For instances, in the following example, `mean_absolute_deviation` acts as an auxiliary function for the target function `find_outlier`.

```
from typing import List

def mean_absolute_deviation(numbers: List[float]) -> float:
    """For a given list of input numbers, calculate Mean Absolute Deviation around the mean of this dataset."""
    mean = sum(numbers) / len(numbers)
    return sum(abs(x - mean) for x in numbers) / len(numbers)

def find_outlier(numbers: list[float]) -> List[float]:
    """For a given list of input numbers, find the outlier. Outliers are defined as data whose distance from the mean is greater than the mean absolute deviation."""
    mean = sum(numbers) / len(numbers)
    mad = mean_absolute_deviation(numbers)
    return [x for x in numbers if abs(x - mean) > mad]
```

### How to evaluate

```
python run_generate.py --model_name codellama/CodeLlama-7b-hf --save_path test
python run_evaluate.py --data_path test/results.parquet --save_path test/execution_results.jsonl
```

## Experimental setup

| Model | Size (B) | Huggingface identifier |
|-------|----------|------------------------|
| InCoder | 1 | `facebook/incoder-1B` |
| InCoder | 6 | `facebook/incoder-6B` |
| CodeGenMulti | 2 | `Salesforce/codegen-2B-multi` |
| CodeGenMulti | 16 | `Salesforce/codegen-16B-multi` |
| CodeGenMono | 2 | `Salesforce/codegen-2B-mono` |
| CodeGenMono | 16 | `Salesforce/codegen-16B-multi` |
| SantaCoder | 1 | `bigcode/santacoder` |
| StarCoder | 16 | `bigcode/starcoder` |
| CodeLLaMA | 7 | `codellama/CodeLlama-7b-hf` |
| CodeLLaMA | 13 | `codellama/CodeLlama-13b-hf` |
| CodeLLaMA | 34 | `codellama/CodeLlama-34b-hf` |
| CodeLLaMAPython | 7 | `codellama/CodeLlama-7b-Python-hf` |
| CodeLLaMAPython | 13 | `codellama/CodeLlama-13b-Python-hf` |
| CodeLLaMAPython | 34 | `codellama/CodeLlama-34b-Python-hf` |
| CodeLLaMAInstruct | 7 | `codellama/CodeLlama-7b-Instruct-hf` |
| CodeLLaMAInstruct | 13 | `codellama/CodeLlama-13b-Instruct-hf` |
| CodeLLaMAInstruct | 34 | `codellama/CodeLlama-34b-Instruct-hf` |

## Warmup. Reproduce HumanEval result

```bash
PYTHONPATH=. python examples/run.py \
  --dataset humaneval \
  --method direct \
  --model facebook/incoder-1B \
  --output_dirpath outputs
```

## Evaluation

```bash
PYTHONPATH=. python examples/run.py \
  --dataset humanextension \
  --method {direct,irrelevant,step_by_step,oracle} \
  --model facebook/incoder-1B \
  --output_dirpath outputs

PYTHONPATH=. python examples/run.py \
  --dataset humaneval \
  --method direct \
  --model google/gemma-2b \
  --output_dirpath outputs \
  --model_api_url http://localhost:8000/v1/completions
```
