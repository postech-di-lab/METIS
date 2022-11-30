import argparse
import logging
import os
import uuid

from transformers import TrainingArguments, set_seed

from clrcmd.data.dataset import (
    ContrastiveLearningCollator,
    NLIContrastiveLearningDataset,
    STSBenchmarkDataset,
)
from clrcmd.data.sts import load_stsb_dev
from clrcmd.models import create_contrastive_learning, create_tokenizer
from clrcmd.trainer import STSTrainer, compute_metrics

logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# fmt: off
parser.add_argument("--data-dir", type=str, help="Data directory", default="data")
parser.add_argument("--model", type=str, help="Model", default="bert-cls",
                    choices=["bert-cls", "bert-avg", "bert-rcmd", "roberta-cls", "roberta-avg", "roberta-rcmd"])
parser.add_argument("--output-dir", type=str, help="Output directory", default="ckpt")
parser.add_argument("--temp", type=float, help="Softmax temperature", default=0.05)
parser.add_argument("--seed", type=int, help="Seed", default=0)
# fmt: on


def main():
    args = parser.parse_args()
    experiment_name = f"{args.model}-{uuid.uuid4()[:6]}"
    training_args = TrainingArguments(
        os.path.join(args.output_dir, experiment_name),
        per_device_train_batch_size=128,
        per_device_eval_batch_size=128,
        learning_rate=5e-5,
        num_train_epochs=3,
        fp16=True,
        logging_strategy="steps",
        logging_steps=20,
        evaluation_strategy="steps",
        eval_steps=250,
        save_strategy="steps",
        save_steps=250,
        metric_for_best_model="eval_spearman",
        load_best_model_at_end=True,
        greater_is_better=True,
        save_total_limit=1,
        seed=args.seed,
    )
    if training_args.local_rank == -1 or training_args.local_rank == 0:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(message)s",
            filename=f"log/train-{experiment_name}.log",
        )
    logger.info("Hyperparameters")
    for k, v in vars(args).items():
        logger.info(f"{k} = {v}")

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, "
        f"device: {training_args.device}, "
        f"n_gpu: {training_args.n_gpu}, "
        f"distributed training: {bool(training_args.local_rank != -1)}, "
        f"16-bits training: {training_args.fp16} "
    )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Load pretrained model and tokenizer
    tokenizer = create_tokenizer(args.model)
    model = create_contrastive_learning(args.model, args.temp)
    model.train()

    train_dataset = NLIContrastiveLearningDataset(
        os.path.join(args.data_dir, "nli_for_simcse.csv"), tokenizer
    )
    eval_dataset = STSBenchmarkDataset(
        load_stsb_dev(os.path.join(args.data_dir, "STS", "STSBenchmark"))["dev"], tokenizer
    )

    trainer = STSTrainer(
        model=model,
        data_collator=ContrastiveLearningCollator(),
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbackes=[],
    )
    train_result = trainer.train()
    logger.info(train_result)
    trainer.save_model(os.path.join(training_args.output_dir, "checkpoint-best"))


if __name__ == "__main__":
    main()
