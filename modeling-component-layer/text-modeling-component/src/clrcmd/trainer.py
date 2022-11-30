from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from scipy.stats import spearmanr
from transformers import EvalPrediction, Trainer
from transformers.utils import logging

logger = logging.get_logger(__name__)


def compute_metrics(x: EvalPrediction) -> Dict[str, float]:
    return {"spearman": spearmanr(x.predictions, x.label_ids).correlation}


class STSTrainer(Trainer):
    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        model.eval()
        with torch.no_grad():
            inputs1 = self._prepare_inputs(inputs["inputs1"])
            inputs2 = self._prepare_inputs(inputs["inputs2"])
            label = self._prepare_inputs(inputs["label"])
            score = model.model(inputs1, inputs2)
        model.train()
        return (None, score, label)
