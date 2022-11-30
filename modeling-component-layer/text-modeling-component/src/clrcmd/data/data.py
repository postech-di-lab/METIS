import itertools
from typing import List, Tuple

Pair = Tuple[int, int]


def create_intervals(tokens: List[str]) -> List[Pair]:
    start_pos = itertools.accumulate(map(len, tokens), initial=0)
    length = map(len, tokens)
    return list(map(lambda x: (x[0], x[0] + x[1]), zip(start_pos, length)))


def is_overlap(interval1: Pair, interval2: Pair) -> bool:
    assert interval1[0] < interval1[1]
    assert interval2[0] < interval2[1]
    state = sorted(
        [(interval1[0], 0), (interval1[1], 1), (interval2[0], 0), (interval2[1], 1)],
        key=lambda x: (x[0], 1 - x[1]),
    )
    state = list(map(lambda x: x[1], state))
    # In this case, only two possible cases are yield in this logic,
    # (0, 0, 1, 1), which is overlapped, or (0, 1, 0, 1), which is exclusive
    return state == [0, 0, 1, 1]


def create_overlap_pairs_from_intervals(
    intervals1: List[Pair], intervals2: List[Pair]
) -> List[Tuple[Pair, Pair]]:
    pipeline = itertools.product(intervals1, intervals2)
    pipeline = filter(lambda x: is_overlap(x[0], x[1]), pipeline)
    return list(pipeline)


def create_perfect_overlap_pairs_from_intervals(
    intervals1: List[Pair], intervals2: List[Pair]
) -> List[Tuple[Pair, Pair]]:
    pipeline = itertools.product(intervals1, intervals2)
    pipeline = filter(lambda x: x[0] == x[1], pipeline)
    return list(pipeline)


def create_perfect_overlap_pairs_from_tokens(
    tokens1: List[str], tokens2: List[str]
) -> List[Tuple[int, int]]:
    intervals1 = create_intervals(tokens1)
    intervals2 = create_intervals(tokens2)
    # NOTE: Due to the special token, the index starts with 1
    interval2idx1 = {x: i for i, x in enumerate(intervals1, start=1)}
    interval2idx2 = {x: i for i, x in enumerate(intervals2, start=1)}
    pairs = create_perfect_overlap_pairs_from_intervals(intervals1, intervals2)
    # Index pair
    pairs = [(interval2idx1[x], interval2idx2[y]) for x, y in pairs]
    return pairs
