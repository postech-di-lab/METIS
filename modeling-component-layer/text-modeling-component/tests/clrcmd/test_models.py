import torch
import torch.nn.functional as F

from clrcmd.models import (
    DensePairwiseRelaxedWordMoverSimilarity,
    PairwiseRelaxedWordMoverSimilarity,
    RelaxedWordMoverSimilarity,
)


def test_pairwise_relaxed_word_mover_similarity():
    torch.manual_seed(0)
    model = PairwiseRelaxedWordMoverSimilarity()

    # Create random input
    x1 = torch.rand((9, 6, 10))
    mask1 = torch.bernoulli(torch.full((x1.shape[0], x1.shape[1]), 0.6)).bool()
    x2 = torch.rand((10, 8, 10))
    mask2 = torch.bernoulli(torch.full((x2.shape[0], x2.shape[1]), 0.6)).bool()

    # Compute similarity using implemented module
    out = model((x1, mask1), (x2, mask2))

    # Naively compute similarity
    sim = F.cosine_similarity(x1[:, None, :, None, :], x2[None, :, None, :, :], dim=-1)
    batchwise_mask1 = mask1[:, None, :, None].expand_as(sim)
    batchwise_mask2 = mask2[None, :, None, :].expand_as(sim)
    sim = torch.where(batchwise_mask1, sim, torch.full_like(sim, float("-inf")))
    sim = torch.where(batchwise_mask2, sim, torch.full_like(sim, float("-inf")))
    sim1 = torch.max(sim, dim=-1)[0]  # (batch1, batch2, seq_len1)
    sim2 = torch.max(sim, dim=-2)[0]  # (batch1, batch2, seq_len2)
    batchwise_mask1 = mask1[:, None, :].expand_as(sim1)
    batchwise_mask2 = mask2[None, :, :].expand_as(sim2)
    sim1 = torch.where(batchwise_mask1, sim1, torch.zeros_like(sim1))
    sim2 = torch.where(batchwise_mask2, sim2, torch.zeros_like(sim2))
    sim1 = torch.sum(sim1, dim=-1) / torch.count_nonzero(batchwise_mask1, dim=-1)
    sim2 = torch.sum(sim2, dim=-1) / torch.count_nonzero(batchwise_mask2, dim=-1)
    sim = (sim1 + sim2) / 2
    assert torch.all(torch.isclose(out, sim))


def test_dense_pairwise_relaxed_word_mover_similarity():
    torch.manual_seed(0)
    model = DensePairwiseRelaxedWordMoverSimilarity()

    # Create random input
    x1 = torch.rand((9, 6, 10))
    mask1 = torch.bernoulli(torch.full((x1.shape[0], x1.shape[1]), 0.6)).bool()
    x2 = torch.rand((10, 8, 10))
    mask2 = torch.bernoulli(torch.full((x2.shape[0], x2.shape[1]), 0.6)).bool()

    # Compute similarity using implemented module
    out = model((x1, mask1), (x2, mask2))

    # Naively compute similarity
    sim = F.cosine_similarity(x1[:, None, :, None, :], x2[None, :, None, :, :], dim=-1)
    batchwise_mask1 = mask1[:, None, :, None].expand_as(sim)
    batchwise_mask2 = mask2[None, :, None, :].expand_as(sim)
    sim = torch.where(batchwise_mask1, sim, torch.full_like(sim, float("-inf")))
    sim = torch.where(batchwise_mask2, sim, torch.full_like(sim, float("-inf")))
    sim1 = torch.max(sim, dim=-1)[0]  # (batch1, batch2, seq_len1)
    sim2 = torch.max(sim, dim=-2)[0]  # (batch1, batch2, seq_len2)
    batchwise_mask1 = mask1[:, None, :].expand_as(sim1)
    batchwise_mask2 = mask2[None, :, :].expand_as(sim2)
    sim1 = torch.where(batchwise_mask1, sim1, torch.zeros_like(sim1))
    sim2 = torch.where(batchwise_mask2, sim2, torch.zeros_like(sim2))
    sim1 = torch.sum(sim1, dim=-1) / torch.count_nonzero(batchwise_mask1, dim=-1)
    sim2 = torch.sum(sim2, dim=-1) / torch.count_nonzero(batchwise_mask2, dim=-1)
    sim = (sim1 + sim2) / 2
    assert torch.all(torch.isclose(out, sim))


def test_relaxed_word_mover_similarity():
    torch.manual_seed(0)
    model = RelaxedWordMoverSimilarity()

    # Create random input
    x1 = torch.rand((2, 6, 10))
    mask1 = torch.bernoulli(torch.full((x1.shape[0], x1.shape[1]), 0.6)).bool()
    x2 = torch.rand((2, 8, 10))
    mask2 = torch.bernoulli(torch.full((x2.shape[0], x2.shape[1]), 0.6)).bool()

    # Compute similarity using implemented module
    out = model((x1, mask1), (x2, mask2))

    # Naively compute similarity
    sim = F.cosine_similarity(x1[:, :, None, :], x2[:, None, :, :], dim=-1)
    batchwise_mask1 = mask1[:, :, None].expand_as(sim)
    batchwise_mask2 = mask2[:, None, :].expand_as(sim)
    sim = torch.where(batchwise_mask1, sim, torch.full_like(sim, float("-inf")))
    sim = torch.where(batchwise_mask2, sim, torch.full_like(sim, float("-inf")))
    sim1 = torch.max(sim, dim=-1)[0]  # (batch, seq_len1)
    sim2 = torch.max(sim, dim=-2)[0]  # (batch, seq_len2)
    sim1 = torch.where(mask1, sim1, torch.zeros_like(sim1))
    sim2 = torch.where(mask2, sim2, torch.zeros_like(sim2))
    sim1 = torch.sum(sim1, dim=-1) / torch.count_nonzero(mask1, dim=-1)
    sim2 = torch.sum(sim2, dim=-1) / torch.count_nonzero(mask2, dim=-1)
    sim = (sim1 + sim2) / 2
    assert torch.all(torch.isclose(out, sim))
