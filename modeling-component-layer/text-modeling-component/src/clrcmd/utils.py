import numpy as np
import torch
from torch import Tensor


def cos(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    u, v = np.nan_to_num(u), np.nan_to_num(v)
    x = np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))
    return np.nan_to_num(x)


def masked_sum(x: Tensor, mask: Tensor, dim: int) -> Tensor:
    """Sum (masked version)

    :param x: Input
    :param mask: Mask Could be broadcastable
    :param dim:
    :return: Result of sum
    """
    return torch.sum(torch.where(mask, x, torch.zeros_like(x)), dim=dim)


def masked_mean(x: Tensor, mask: Tensor, dim: int) -> Tensor:
    """Mean (masked version)

    :param x: Input
    :param mask: Mask Could be broadcastable
    :param dim:
    :return: Result of mean
    """
    return masked_sum(x, mask, dim) / torch.count_nonzero(mask, dim=dim)
