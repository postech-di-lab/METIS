import logging
from typing import Dict, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from transformers import (
    AutoModel,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from clrcmd.utils import masked_mean

logger = logging.getLogger(__name__)


def dist_all_gather(x: Tensor) -> Tensor:
    """Boilerplate code for all gather in distributed setting

    The first dimension could be different

    :param x: Tensor to be gathered
    :type x: Tensor
    :return: Tensor after gathered. For the gradient flow, current rank is
             replaced to original tensor
    :rtype: Tensor
    """
    assert dist.is_initialized(), "The process is not in DDP setting"
    world_size = dist.get_world_size()
    # 1. Get size acroess processes
    x_numel_list = [torch.tensor(x.numel(), device=x.device) for _ in range(world_size)]
    dist.all_gather(x_numel_list, torch.tensor(x.numel(), device=x.device))
    # 2. Infer maximum size
    max_size = max(x.item() for x in x_numel_list)
    # 3. Communitcate tensor with padded version
    _x_list = [torch.empty((max_size,), device=x.device) for _ in range(world_size)]
    _x = torch.cat(
        (
            x.contiguous().view(-1),
            torch.empty((max_size - x.numel(),), device=x.device),
        )
    )
    dist.all_gather(_x_list, _x)
    # 4. Remove padded data to change original shape
    x_list = [_x[:n].view(-1, *x.shape[1:]) for n, _x in zip(x_numel_list, _x_list)]
    # Since `all_gather` results do not have gradients, we replace the
    # current process's corresponding embeddings with original tensors
    x_list[dist.get_rank()] = x
    return torch.cat(x_list, dim=0)


ModelInput = Dict[str, Tensor]


class LastHiddenSentenceRepresentationModel(nn.Module):
    def __init__(self, model: PreTrainedModel, head: bool = False):
        super().__init__()
        self.model = model
        self.head = head
        if head:
            hidden_size = self.model.config.hidden_size
            self.linear = nn.Linear(hidden_size, hidden_size)

    def forward(self, inputs: ModelInput) -> Tuple[Tensor, Tensor]:
        # Return representation with mask
        outputs = self.model(**inputs).last_hidden_state
        if self.head:
            outputs = self.linear(outputs)
        return outputs, inputs["attention_mask"].bool()

    def compute_last_hidden(self, inputs: ModelInput) -> Tuple[Tensor, Tensor]:
        return self.forward(inputs)


class CLSPoolingSentenceRepresentationModel(nn.Module):
    def __init__(self, model: PreTrainedModel, head: bool = False):
        super().__init__()
        self.model = model
        self.head = head
        if head:
            hidden_size = self.model.config.hidden_size
            self.linear = nn.Linear(hidden_size, hidden_size)

    def forward(self, inputs: ModelInput) -> Tensor:
        outputs = self.model(**inputs).last_hidden_state[:, 0]
        if self.head:
            outputs = self.linear(outputs)
        return outputs

    def compute_last_hidden(self, inputs: ModelInput) -> Tuple[Tensor, Tensor]:
        raise ValueError("No interpretable resource for cls pooling")


class AveragePoolingSentenceRepresentationModel(nn.Module):
    def __init__(self, model: PreTrainedModel, head: bool = False):
        super().__init__()
        self.model = model
        self.head = head
        if head:
            hidden_size = self.model.config.hidden_size
            self.linear = nn.Linear(hidden_size, hidden_size)

    def forward(self, inputs: ModelInput) -> Tensor:
        mask = inputs["attention_mask"].bool().unsqueeze(2)
        outputs = masked_mean(self.model(**inputs).last_hidden_state, mask, dim=1)
        if self.head:
            outputs = self.linear(outputs)
        return outputs

    def compute_last_hidden(self, inputs: ModelInput) -> Tuple[Tensor, Tensor]:
        outputs = self.model(**inputs).last_hidden_state
        if self.head:
            outputs = self.linear(outputs)
        return outputs, inputs["attention_mask"].bool()


class SentenceBertLearningModule(nn.Module):
    def __init__(self, model: nn.Module, hidden_size: int):
        super().__init__()
        self.model = model
        self.head = nn.Linear(hidden_size, 3, bias=False)  # 3-way classification
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, inputs1: ModelInput, inputs2: ModelInput, labels: Tensor) -> Tuple[Tensor]:
        x1, x2 = self.representation_model(inputs1), self.representation_model(inputs2)
        pred = self.head(torch.cat((x1, x2, torch.abs(x1 - x2)), dim=2))
        return (self.criterion(pred, labels),)


class SentenceSimilarityModel(nn.Module):
    def __init__(self, representation_model: nn.Module, similarity: nn.Module):
        super().__init__()
        self.representation_model = representation_model
        self.similarity = similarity

    def forward(self, inputs1: ModelInput, inputs2: ModelInput) -> Tensor:
        """Provide similarity between two sentences

        :param inputs1: model input for sentence1.
        :param inputs2: model input for sentence2.
        :return: similarity score.
        """
        x1, x2 = self.representation_model(inputs1), self.representation_model(inputs2)
        return self.similarity(x1, x2)

    def compute_heatmap(self, inputs1: ModelInput, inputs2: ModelInput) -> Tensor:
        x1 = self.representation_model.compute_last_hidden(inputs1)
        x2 = self.representation_model.compute_last_hidden(inputs2)
        return self.similarity.compute_heatmap(x1, x2)


class SimcseLearningModule(nn.Module):
    def __init__(self, model: nn.Module, pairwise_similarity: nn.Module, temp: float):
        super().__init__()
        self.model = model
        self.pairwise_similarity = pairwise_similarity
        self.temp = temp
        self.criterion = nn.CrossEntropyLoss()

    def forward(
        self,
        inputs1: ModelInput,
        inputs2: ModelInput,
        inputs_neg: Optional[ModelInput] = None,
    ) -> Tuple[Tensor]:
        if inputs_neg is not None:
            inputs = {
                k: torch.cat((inputs1[k], inputs2[k], inputs_neg[k]), dim=0)
                for k in inputs1.keys()
            }
        else:
            inputs = {k: torch.cat((inputs1[k], inputs2[k]), dim=0) for k in inputs1.keys()}
        x = self.model.representation_model(inputs)
        if inputs_neg is not None:
            sections = (
                inputs1["input_ids"].shape[0],
                inputs2["input_ids"].shape[0] + inputs_neg["input_ids"].shape[0],
            )
        else:
            sections = inputs1["input_ids"].shape[0], inputs2["input_ids"].shape[0]

        # NOTE: Really bad.. how to fix it?
        if type(x) == Tensor:
            x1, x2 = torch.split(x, sections)
        else:
            x1, x2 = list(zip(torch.split(x[0], sections), torch.split(x[1], sections)))
        sim = self.pairwise_similarity(x1, x2)
        sim = sim / self.temp
        # (batch_size, batch_size)
        labels = torch.arange(sim.shape[0], dtype=torch.long, device=sim.device)
        return (self.criterion(sim, labels),)


def compute_alignment(
    x1: Tensor, x2: Tensor, mask1: Tensor, mask2: Tensor
) -> Tuple[Tensor, Tensor]:
    sim = F.cosine_similarity(x1.unsqueeze(-2), x2.unsqueeze(-3), dim=-1)
    # Set similarity of invalid position to negative inf
    inf = torch.tensor(float("-inf"), device=sim.device)
    sim = torch.where(mask1.unsqueeze(-1), sim, inf)
    sim = torch.where(mask2.unsqueeze(-2), sim, inf)
    indice1 = torch.max(sim, dim=-1)[1]
    indice2 = torch.max(sim, dim=-2)[1]
    return indice1, indice2


class RelaxedWordMoverSimilarity(nn.Module):
    def __init__(self):
        super().__init__()
        self.cos = nn.CosineSimilarity(dim=-1)

    def forward(self, x1: Tuple[Tensor, Tensor], x2: Tuple[Tensor, Tensor]) -> Tensor:
        """Compute relaxed word mover similarity

        :param x1: ((batch, seq_len1, hidden_dim), (batch, seq_len1)), torch.float
        :param x2: ((batch, seq_len2, hidden_dim), (batch, seq_len2)), torch.float
        :return: (batch)
        """
        (x1, mask1), (x2, mask2) = x1, x2
        sim = self.cos(x1[:, :, None, :], x2[:, None, :, :])
        inf = torch.tensor(float("-inf"), device=sim.device)
        sim = torch.where(mask1.unsqueeze(-1), sim, inf)
        sim = torch.where(mask2.unsqueeze(-2), sim, inf)
        # (batch, seq_len1, seq_len2)
        sim1, sim2 = torch.max(sim, dim=2)[0], torch.max(sim, dim=1)[0]
        sim1 = masked_mean(sim1, mask1, dim=1)
        sim2 = masked_mean(sim2, mask2, dim=1)
        sim = (sim1 + sim2) / 2
        return sim

    def compute_heatmap(self, x1: Tuple[Tensor, Tensor], x2: Tuple[Tensor, Tensor]) -> Tensor:
        (x1, mask1), (x2, mask2) = x1, x2
        sim = self.cos(x1[:, :, None, :], x2[:, None, :, :])
        inf = torch.tensor(float("-inf"), device=sim.device)
        sim = torch.where(mask1.unsqueeze(-1), sim, inf)
        sim = torch.where(mask2.unsqueeze(-2), sim, inf)
        # (batch, seq_len1, seq_len2)
        sim1 = torch.mul(sim, (sim == torch.max(sim, dim=2, keepdim=True)[0]).float())
        sim2 = torch.mul(sim, (sim == torch.max(sim, dim=1, keepdim=True)[0]).float())
        sim1 = sim1 / torch.count_nonzero(mask1, dim=1)[:, None, None]
        sim2 = sim2 / torch.count_nonzero(mask2, dim=1)[:, None, None]
        sim = (sim1 + sim2) / 2
        return sim


class PairwiseRelaxedWordMoverSimilarity(nn.Module):
    def __init__(self):
        super().__init__()
        self.cos = nn.CosineSimilarity(dim=-1)

    def forward(self, x1: Tuple[Tensor, Tensor], x2: Tuple[Tensor, Tensor]) -> Tensor:
        """Compute relaxed word mover similarity

        :param x1: ((batch1, seq_len1, hidden_dim), (batch1, seq_len1)), torch.float
        :param x2: ((batch2, seq_len2, hidden_dim), (batch2, seq_len2)), torch.float
        :return: (batch1, batch2)
        """
        (x1, mask1), (x2, mask2) = x1, x2
        batch1, seq_len1, hidden_dim = x1.shape
        batch2, seq_len2, _ = x2.shape
        # Compute max indice batchwise
        with torch.no_grad():
            indice1 = torch.empty((batch1, batch2, seq_len1), dtype=torch.long, device=x1.device)
            indice2 = torch.empty((batch1, batch2, seq_len2), dtype=torch.long, device=x2.device)
            for i in range(0, batch1, 8):
                for j in range(0, batch2, 8):
                    _indice1, _indice2 = compute_alignment(
                        x1[i : i + 8, None, :, :],
                        x2[None, j : j + 8, :, :],
                        mask1[i : i + 8, None, :],
                        mask2[None, j : j + 8, :],
                    )
                    indice1[i : i + 8, j : j + 8, :] = _indice1
                    indice2[i : i + 8, j : j + 8, :] = _indice2
        # Construct computational graph for RWMD
        x1, x2 = x1.unsqueeze(1), x2.unsqueeze(0)
        sim1 = self.cos(
            x1,  # (batch1, 1, seq_len1, hidden_dim)
            torch.gather(
                x2.expand((batch1, -1, -1, -1)),
                dim=2,
                index=indice1.unsqueeze(-1).expand((-1, -1, -1, hidden_dim)),
            ),  # (batch1, batch2, seq_len1, hidden_dim)
        )
        # (batch1, batch2, seq_len1)
        sim2 = self.cos(
            torch.gather(
                x1.expand((-1, batch2, -1, -1)),
                dim=2,
                index=indice2.unsqueeze(-1).expand((-1, -1, -1, hidden_dim)),
            ),  # (batch1, batch2, seq_len2, hidden_dim)
            x2,  # (1, batch2, seq_len2, hidden_dim)
        )
        # (batch1, batch2, seq_len2)
        sim1 = masked_mean(sim1, mask1.unsqueeze(1).expand_as(sim1), dim=-1)
        sim2 = masked_mean(sim2, mask2.unsqueeze(0).expand_as(sim2), dim=-1)
        sim = (sim1 + sim2) / 2
        return sim


class DensePairwiseRelaxedWordMoverSimilarity(nn.Module):
    def __init__(self):
        super().__init__()
        self.cos = nn.CosineSimilarity(dim=-1)

    def forward(self, x1: Tuple[Tensor, Tensor], x2: Tuple[Tensor, Tensor]) -> Tensor:
        """Compute relaxed word mover similarity

        :param x1: ((batch1, seq_len1, hidden_dim), (batch1, seq_len1)), torch.float
        :param x2: ((batch2, seq_len2, hidden_dim), (batch2, seq_len2)), torch.float
        :return: (batch1, batch2)
        """
        (x1, mask1), (x2, mask2) = x1, x2
        sim = self.cos(x1[:, None, :, None, :], x2[None, :, None, :, :])
        # (batch1, batch2, seq_len1, seq_len2)
        inf = torch.tensor(float("-inf"), device=sim.device)
        sim = torch.where(mask1[:, None, :, None], sim, inf)
        sim = torch.where(mask2[None, :, None, :], sim, inf)
        sim1, sim2 = torch.max(sim, dim=3)[0], torch.max(sim, dim=2)[0]
        # (batch1, batch2, seq_len1), (batch1, batch2, seq_len2)
        sim1 = masked_mean(sim1, mask1[:, None, :], dim=-1)
        sim2 = masked_mean(sim2, mask2[None, :, :], dim=-1)
        sim = (sim1 + sim2) / 2
        return sim


class CosineSimilarity(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.cos = nn.CosineSimilarity(dim=dim)

    def forward(self, x1: Tensor, x2: Tensor) -> Tensor:
        return self.cos(x1, x2)

    def compute_heatmap(self, x1: Tuple[Tensor, Tensor], x2: Tuple[Tensor, Tensor]) -> Tensor:
        (x1, mask1), (x2, mask2) = x1, x2
        s1 = masked_mean(x1, mask1.unsqueeze(2), dim=1)  # (batch, hidden)
        s2 = masked_mean(x2, mask2.unsqueeze(2), dim=1)  # (batch, hidden)
        sim = torch.einsum("bih,bjh->bij", x1, x2)
        inf = torch.tensor(float("-inf"), device=sim.device)
        sim = torch.where(mask1.unsqueeze(-1), sim, inf)
        sim = torch.where(mask2.unsqueeze(-2), sim, inf)
        sim = sim / torch.norm(s1, dim=1)[:, None, None]
        sim = sim / torch.norm(s2, dim=1)[:, None, None]
        sim = sim / torch.count_nonzero(mask1, dim=1)[:, None, None]
        sim = sim / torch.count_nonzero(mask2, dim=1)[:, None, None]
        return sim


class PairwiseCosineSimilarity(nn.Module):
    def forward(self, x1: Tensor, x2: Tensor) -> Tensor:
        return F.cosine_similarity(x1.unsqueeze(1), x2.unsqueeze(0), dim=-1)


def create_tokenizer(model_name: str) -> PreTrainedTokenizerBase:
    if model_name.startswith("bert"):
        return AutoTokenizer.from_pretrained("bert-base-uncased", use_fast=False)
    elif model_name.startswith("roberta"):
        return AutoTokenizer.from_pretrained("roberta-base", use_fast=False)
    else:
        raise ValueError(f"Undefined {model_name = }")


def create_similarity_model(model_name: str) -> nn.Module:
    if model_name.startswith("bert"):
        model = AutoModel.from_pretrained("bert-base-uncased")
    elif model_name.startswith("roberta"):
        model = AutoModel.from_pretrained("roberta-base")
    else:
        raise ValueError(f"Undefined {model_name = }")
    if model_name.endswith("cls"):
        model = CLSPoolingSentenceRepresentationModel(model, head=True)
        model = SentenceSimilarityModel(model, CosineSimilarity(dim=-1))
    elif model_name.endswith("avg"):
        model = AveragePoolingSentenceRepresentationModel(model, head=True)
        model = SentenceSimilarityModel(model, CosineSimilarity(dim=-1))
    elif model_name.endswith("rcmd"):
        model = LastHiddenSentenceRepresentationModel(model, head=True)
        model = SentenceSimilarityModel(model, RelaxedWordMoverSimilarity())
    else:
        raise ValueError(f"Undefined {model_name = }")
    return model


def create_contrastive_learning(
    model_name: str, temp: float = 1.0, dense_rwmd: bool = False
) -> nn.Module:
    model = create_similarity_model(model_name)
    if model_name.endswith("cls"):
        return SimcseLearningModule(model, PairwiseCosineSimilarity(), temp)
    elif model_name.endswith("avg"):
        return SimcseLearningModule(model, PairwiseCosineSimilarity(), temp)
    elif model_name.endswith("rcmd"):
        if dense_rwmd:
            pairwise_similarity = DensePairwiseRelaxedWordMoverSimilarity()
        else:
            pairwise_similarity = PairwiseRelaxedWordMoverSimilarity()
        return SimcseLearningModule(model, pairwise_similarity, temp)
    else:
        raise ValueError(f"Undefined {model_name = }")
