import ast

from loguru import logger
import numpy as np

from humanextension.dataset import load_humaneval_multiple, load_humaneval_openai


def parse_function_name(code: str) -> str:
    return ast.parse(code).body[0].name


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


def parse_docstring(code: str, clean: bool=False) -> str:
    node = ast.parse(code)
    for segment in node.body:
        if isinstance(segment, ast.FunctionDef):
            docstring = ast.get_docstring(segment, clean=clean)
            return docstring
    raise ValueError("Function signature not found")


def replace_function_name(code: str, replace_name: str) -> str:
    node = ast.parse(code).body[0]
    node.name = replace_name
    return ast.unparse(node)


def replace_function_docstring(code: str, replace_docstring: str) -> str:
    node = ast.parse(code).body[0]
    # NOTE: We assume that docstring is always in the code
    node.body[0].value.value = replace_docstring
    return ast.unparse(node)


def remove_function_docstring(code: str) -> str:
    node = ast.parse(code).body[0]
    if ast.get_docstring(node) is None:
        logger.warning("No docstring found.. Something is weird.")
    node.body = node.body[1:]
    return ast.unparse(node)


def create_humaneval_functions() -> list[tuple[tuple[str, ...], str]]:
    dataset_multiple = load_humaneval_multiple()
    dataset_openai = load_humaneval_openai()

    def create_identifier_multipl(row) -> str:
        return f"{row['task_id'].replace('/', '_')}_{row['entry_point']}"

    dataset_openai["identifier_multipl"] = dataset_openai.apply(
        create_identifier_multipl, axis=1)
    dataset = dataset_openai.merge(dataset_multiple,
                                   left_on="identifier_multipl",
                                   right_on="name")

    def create_implementation(row) -> str:
        return f"{row['prompt_y']}{row['canonical_solution']}"

    dataset["implementation"] = dataset.apply(create_implementation, axis=1)

    def parse_implementation(row) -> tuple[tuple[str, ...], str]:
        implementation = row["implementation"]
        node = ast.parse(implementation)

        import_statements = []
        function_implementation = None
        for n in node.body:
            if isinstance(n, ast.ImportFrom):
                import_statements.append(
                    ast.get_source_segment(implementation, n))
            elif isinstance(n, ast.Import):
                import_statements.append(
                    ast.get_source_segment(implementation, n))
            elif isinstance(n, ast.FunctionDef):
                assert function_implementation is None
                function_implementation = ast.get_source_segment(
                    implementation, n)
            else:
                raise ValueError()
        return import_statements, function_implementation

    return dataset.apply(parse_implementation, axis=1).tolist()


def remove_after_stop_token(completion: str, prompt: str,
                            stop_tokens: list[str]) -> str:
    min_stop_index = len(completion)
    for stop_token in stop_tokens:
        stop_index = completion.find(stop_token, len(prompt))
        if stop_index != -1 and stop_index < min_stop_index:
            min_stop_index = stop_index
    return completion[:min_stop_index]


def estimate_pass_at_k(num_samples: int, num_corrects: int, k: int) -> float:
    """각 문제에 대한 pass@k를 추정하고 이를 배열로 반환합니다.
    pass@k를 추정합니다. n개의 구현 중에 c개의 구현이 맞을 경우, n-c개의 틀린 구현들로만 k개가 생성되었을 경우의 수 대비
    n개의 구현 중 k개가 생성될 경우의 수를 확률로 삼아, 적어도 하나의 구현이 맞을 확률을 추정합니다.
    이는, 1 - comb(n - c, k) / comb(n, k)로 표현될 수 있습니다.
    만약 n-c개의 틀린 구현들로 k개가 되지 않을 경우, 반드시 맞는 구현이 포함되므로 -1.0을 반환합니다.

    :params num_samples: 각 문제에 대한 샘플의 수
    :params num_corrects: 각 문제에 대한 정답의 수
    :params k: pass@k를 계산할 k
    :returns: 각 문제에 대한 pass@k
    """
    assert num_corrects >= 0
    assert num_samples >= 0
    if num_corrects == 0 or num_samples == 0:
        return 0.0
    if num_samples - num_corrects < k:
        return 1.0
    return 1.0 - np.prod(
        1.0 - k / np.arange(num_samples - num_corrects + 1, num_samples + 1))
