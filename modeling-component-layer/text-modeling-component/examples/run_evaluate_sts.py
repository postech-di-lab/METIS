import argparse
import logging
import os

import numpy as np
import torch
from scipy.stats import spearmanr
from torch.utils.data import DataLoader
from tqdm import tqdm

from clrcmd.data.dataset import STSBenchmarkDataset
from clrcmd.data.sts import load_sts_benchmark
from clrcmd.models import create_contrastive_learning, create_tokenizer

logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# fmt: off
parser.add_argument("--model", type=str, help="Model", default="bert-cls",
                    choices=["bert-cls", "bert-avg", "bert-rcmd", "roberta-cls", "roberta-avg", "roberta-rcmd"])
parser.add_argument("--checkpoint", type=str, help="Checkpoint path", default=None)
parser.add_argument("--data-dir", type=str, help="data dir", default="data")
# fmt: on


def main():
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    os.makedirs("log", exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(message)s",
        filename=f"log/evaluate-{args.model}.log",
    )
    logger.info("** Command Line Arguments **")
    for k, v in vars(args).items():
        logger.info(f"  {k}: {v}")

    # Create tokenizer and model
    tokenizer = create_tokenizer(args.model)
    model = create_contrastive_learning(args.model).to(device)

    # Load method
    if args.checkpoint is not None:
        model.load_state_dict(torch.load(os.path.join(args.checkpoint, "pytorch_model.bin")))
    model = model.model

    # Load dataset
    sources = load_sts_benchmark(args.data_dir)
    loaders = {
        name: {
            k: DataLoader(STSBenchmarkDataset(v, tokenizer), batch_size=32)
            for k, v in testset.items()
        }
        for name, testset in sources.items()
    }

    # Evaluate
    result = {}
    model.eval()
    with torch.no_grad():
        for source_name, source in loaders.items():
            logger.info(f"Evaluate {source_name}")
            scores_all, labels_all = [], []
            for _, loader in source.items():
                scores, labels = [], []
                for examples in tqdm(loader, desc=f"Evaluate {source_name}"):
                    inputs1 = {k: v.to(device) for k, v in examples["inputs1"].items()}
                    inputs2 = {k: v.to(device) for k, v in examples["inputs2"].items()}
                    scores.append(model(inputs1, inputs2).cpu().numpy())
                    labels.append(examples["label"].numpy())
                scores, labels = np.concatenate(scores), np.concatenate(labels)
                scores_all.append(scores)
                labels_all.append(labels)
            scores_all, labels_all = np.concatenate(scores_all), np.concatenate(labels_all)
            result[source_name] = spearmanr(scores_all, labels_all)[0]

    logger.info("** Result **")
    for metric_name, metric_value in result.items():
        logger.info(f"{metric_name} = {metric_value:.4f}")
    score_avg = np.average(list(result.values()))
    logger.info(f"avg = {score_avg:.4f}")


if __name__ == "__main__":
    main()
