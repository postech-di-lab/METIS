import sys
import json
import logging
import resource
import os
import asyncio
import tempfile
import argparse
import queue
import re
import dataclasses
from collections import defaultdict


def set_memory_limit(limit_mb):
    # CPU limit
    resource.setrlimit(resource.RLIMIT_CPU, (10, 10))

    # Memory limit
    limit_bytes = limit_mb * 1024 * 1024
    resource.setrlimit(resource.RLIMIT_AS, (limit_bytes, limit_bytes))

def preexec():
    set_memory_limit(4096)

@dataclasses.dataclass(frozen=True)
class ExecutionInput:
    code: str
    stdin: list[str]

@dataclasses.dataclass(frozen=True)
class ExecutionOutput:
    stdout: list[str]
    stderr: list[str]
    returncode: list[int]

@dataclasses.dataclass(frozen=True)
class PredictionRecord:
    index: str
    problem_id: int
    code: str

@dataclasses.dataclass(frozen=True)
class ResultRecord:
    index: str
    problem_id: int
    stdout: list[str]
    stderr: list[str]
    returncode: list[int]

async def execute_python3(input: ExecutionInput) -> ExecutionOutput:
    os.makedirs("temp", exist_ok=True)
    with tempfile.TemporaryDirectory(dir="temp") as dirname:
        filepath = os.path.join(os.path.abspath(dirname), "solution.py")
        with open(filepath, 'w') as f:
            sol = "import sys\nimport time\nimport itertools\nfrom itertools import accumulate, product, permutations, combinations\nimport collections\nfrom collections import Counter, OrderedDict, deque, defaultdict, ChainMap\nfrom functools import lru_cache\nimport math\nfrom math import sqrt, sin, cos, tan, ceil, fabs, floor, gcd, exp, log, log2\nimport fractions\nfrom typing import *\nimport numpy as np\nimport random\nimport heapq\nfrom heapq import *\nstdin = sys.stdin\nstdout = sys.stdout\n"
            f.write(sol + input.code)
        
        # Early exit. Sequentially execute testcase
        total_stdout, total_stderr, total_returncode = [], [], []
        for stdin in input.stdin:
            proc = await asyncio.create_subprocess_shell(
                'python3 solution.py',
                cwd=dirname,
                preexec_fn=preexec,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE)
            stdout, stderr = await proc.communicate(stdin.encode())
            total_stdout.append(stdout.decode())
            total_stderr.append(stderr.decode())
            total_returncode.append(proc.returncode)
        return ExecutionOutput(
            stdout=total_stdout,
            stderr=total_stderr,
            returncode=total_returncode
        )

async def process_prediction(
        worker_idx: int,
        queue_in: queue.Queue[PredictionRecord | None],
        queue_out: queue.Queue[ResultRecord | None]):
    while True:
        prediction: PredictionRecord | None = await queue_in.get()
        if prediction is None:
            await queue_out.put(None)
            queue_in.task_done()
            break
        logging.info(f"worker {worker_idx} - {prediction.index} - {prediction.problem_id}")

        execution_input = ExecutionInput(code=prediction.code, stdin=[""])
        coroutine = asyncio.create_task(execute_python3(execution_input))
        execution_output = await coroutine

        result = ResultRecord(
            index=prediction.index,
            problem_id=prediction.problem_id,
            stdout=execution_output.stdout,
            stderr=execution_output.stderr,
            returncode=execution_output.returncode,
        )
        await queue_out.put(result)
        queue_in.task_done()

async def process_predictions(predictions: list[PredictionRecord],
                              max_workers: int) -> list[ResultRecord]:
    logging.info("Create queue")
    queue_in, queue_out = asyncio.Queue(maxsize=1024), asyncio.Queue()

    logging.info("Create worker")
    workers = [asyncio.create_task(process_prediction(idx, queue_in, queue_out))
               for idx in range(max_workers)]
    for idx, prediction in enumerate(predictions):
        if idx % 20 == 0:
            logging.info(f"{idx}/{len(predictions)} submitted.")
        await queue_in.put(prediction)
    for _ in workers:
        await queue_in.put(None)

    logging.info("Join queue and worker")
    await queue_in.join()
    await asyncio.gather(*workers)

    logging.info("Collect outputs")
    results = []
    while not queue_out.empty():
        result: ResultRecord | None = await queue_out.get()
        if result is not None:
            results.append(result)
        queue_out.task_done()
    return results

parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, required=True)
parser.add_argument("--save_path", type=str, required=True)
parser.add_argument("--max_workers", type=int, default=32)

import pandas as pd

async def main():
    logging.basicConfig(level=logging.INFO, stream=sys.stdout, format="%(asctime)s:%(message)s")
    args = parser.parse_args()

    logging.info(f"Load data")
    df = pd.read_parquet(args.data_path)
    
    def create_code(row: dict):
        if row["code_flexible"] is not None:
            return row['function1_human'] + "\n\n\n" + row["code_flexible"] + "\n\n\n" + row["tests"]
        else:
            return "assert False"

    predictions = [
        PredictionRecord(index=idx,
                         problem_id=row['problem_number'],
                         code=create_code(row))
        for idx, row in df.iterrows()
    ]

    results = await process_predictions(predictions, args.max_workers)

    logging.info("Compute metric")
    metrics = defaultdict(list)
    for result in results:
        metrics[result.problem_id].append(result.returncode[0] == 0)
    pass_scores = []
    for _, corrects in metrics.items():
        correct, total = sum(corrects), len(corrects)
        pass_scores.append(correct / total)  # pass@1
    pass_score = sum(pass_scores) / len(pass_scores)
    print(f"Pass@1: {pass_score}")

    import pdb; pdb.set_trace()

    logging.info("Save results")
    if os.path.dirname(args.save_path):
        os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    with open(args.save_path, 'w') as f:
        f.writelines(json.dumps(dataclasses.asdict(result)) + "\n" for result in results)


if __name__ == "__main__":
    asyncio.run(main())