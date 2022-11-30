import argparse
import json
import logging
import os
from typing import Tuple

import torch
from transformers import AutoTokenizer

from clrcmd.evaluation.ists import inference, load_examples, preprocess, save
from clrcmd.models import create_contrastive_learning

# fmt: off
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--data-dir", type=str, required=True, help="data dir")
parser.add_argument("--source", type=str, required=True, choices=["images", "headlines", "answers-students"], help="source")
parser.add_argument("--checkpoint-dir", type=str, required=True, help="checkpoint directory")
# fmt: on


def create_filepaths(data_dir: str, source: str) -> Tuple[str, str, str, str]:
    return (
        os.path.join(data_dir, f"STSint.testinput.{source}.sent1.txt"),
        os.path.join(data_dir, f"STSint.testinput.{source}.sent2.txt"),
        os.path.join(data_dir, f"STSint.testinput.{source}.sent1.chunk.txt"),
        os.path.join(data_dir, f"STSint.testinput.{source}.sent2.chunk.txt"),
    )


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s\t%(levelname)s\t%(message)s")
    args = parser.parse_args()

    examples = load_examples(*create_filepaths(args.data_dir, args.source))
    logging.info(f"Loading iSTS example (source = {args.source})")
    logging.info(f"{examples[0] = }")

    with open(os.path.join(args.checkpoint_dir, "model_args.json")) as f:
        model_args = json.load(f)
    logging.info("Load model configuration")
    logging.info(f"{model_args = }")
    tokenizer = AutoTokenizer.from_pretrained(model_args["huggingface_model_name"], use_fast=False)
    logging.info(f"Loading tokenizer (model = {model_args['huggingface_model_name']})")
    examples = preprocess(tokenizer=tokenizer, examples=examples)
    logging.info("Preprocess examples (Tokenize examples)")
    logging.info(f"{examples[0] = }")

    module = create_contrastive_learning(
        model_name=model_args["model_name"], temp=model_args["temp"], dense_rwmd=False
    )
    if os.path.exists(os.path.join(args.checkpoint_dir, "pytorch_model.bin")):
        module.load_state_dict(torch.load(os.path.join(args.checkpoint_dir, "pytorch_model.bin")))
        logging.info("Load model")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    examples = inference(model=module.model, prep_examples=examples, device=device)
    logging.info("Perform inference")
    logging.info(f"{examples[0] = }")

    outfile = f"{args.source}.wa" if args.checkpoint_dir else f"{args.source}.wa.untrained"
    save(examples, os.path.join(args.checkpoint_dir, outfile))
    logging.info("Complete saving examples")
    exit()


if __name__ == "__main__":
    main()
