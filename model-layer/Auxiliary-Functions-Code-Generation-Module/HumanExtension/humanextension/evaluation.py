from concurrent.futures import ProcessPoolExecutor, as_completed

from humanextension.execution import check_correctness


def evaluate_functional_correctness(row: dict) -> list[str]:
    # Create program for verifying functional correctness
    programs = [
        f"{implementation}\n{row['tests']}"
        for implementation in row["implementations"]
    ]
    with ProcessPoolExecutor(max_workers=4) as executor:
        futures = []
        for idx, program in enumerate(programs):
            futures.append(
                executor.submit(check_correctness, idx, program, 3.0))
        results = [None for _ in range(len(programs))]
        for future in as_completed(futures):
            idx, result = future.result()
            results[idx] = result
    assert all(result is not None for result in results)
    return results
