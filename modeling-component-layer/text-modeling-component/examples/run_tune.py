import logging
import os

import optuna
from optuna import Trial
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


def objective(trial: Trial):
    experiment_name = f"{trial.study.study_name}-{trial.number}"
    training_args = TrainingArguments(
        os.path.join("ckpt", experiment_name),
        per_device_train_batch_size=128,
        per_device_eval_batch_size=128,
        learning_rate=trial.suggest_categorical("learning_rate", [3e-5, 4e-5, 5e-5, 6e-5, 7e-5]),
        num_train_epochs=3,
        fp16=True,
        logging_strategy="steps",
        logging_steps=20,
        evaluation_strategy="steps",
        eval_steps=250,
        metric_for_best_model="eval_spearman",
        load_best_model_at_end=True,
        greater_is_better=True,
        save_total_limit=1,
        seed=trial.suggest_categorical("seed", [2, 3, 4]),
    )
    if training_args.local_rank == -1 or training_args.local_rank == 0:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(message)s",
            filename=f"log/train-{experiment_name}.log",
        )

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
    tokenizer = create_tokenizer("bert-rcmd")
    model = create_contrastive_learning(
        "bert-rcmd", trial.suggest_categorical("temp", [0.025, 0.05, 0.075])
    )
    model.train()

    train_dataset = NLIContrastiveLearningDataset(
        os.path.join("data", "nli_for_simcse.csv"), tokenizer
    )
    eval_dataset = STSBenchmarkDataset(
        load_stsb_dev(os.path.join("data", "STS", "STSBenchmark"))["dev"], tokenizer
    )

    trainer = STSTrainer(
        model=model,
        data_collator=ContrastiveLearningCollator(),
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )
    train_result = trainer.train()
    logger.info(train_result)
    trainer.save_model(os.path.join(training_args.output_dir, "checkpoint-best"))
    return trainer.evaluate()["eval_spearman"]


def main():
    study = optuna.create_study(
        study_name="tune",
        direction="maximize",
        storage="sqlite:///tune.db",
        sampler=optuna.samplers.TPESampler(),
        pruner=optuna.pruners.MedianPruner(),
        load_if_exists=True,
    )
    study.optimize(objective, n_trials=10)
    print(f"{study.best_params = }")
    print(f"{study.best_value = }")
    print(f"{study.best_trial = }")


if __name__ == "__main__":
    main()
