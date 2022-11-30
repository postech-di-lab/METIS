import argparse
import logging

import torch
from sentsim.config import ModelArguments
from sentsim.models.models import create_similarity_model

logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
# fmt: off
parser.add_argument("--batch-size", type=int, default=512)
parser.add_argument("--seq-len", type=int, default=8)
parser.add_argument("--method", type=str, default="rwmdcse", choices=["simcse-cls", "simcse-avg", "rwmdcse"])
parser.add_argument("--num-iter", type=int, default=10)
# fmt: on


def main(args):
    # Create timer
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    # Identify device
    device = torch.device("cuda")

    # Generate sample
    sent1 = {
        "input_ids": torch.randint(0, 1000, (args.batch_size, args.seq_len)).to(device),
        "attention_mask": torch.ones(args.batch_size, args.seq_len).to(device),
    }
    sent2 = {
        "input_ids": torch.randint(0, 1000, (args.batch_size, args.seq_len)).to(device),
        "attention_mask": torch.ones(args.batch_size, args.seq_len).to(device),
    }

    # Build model
    model_args = ModelArguments(loss_type=args.method)
    model = create_similarity_model(model_args).to(device)

    # Execution
    with torch.no_grad():
        # Warmup
        _ = model(sent1, sent2)
        start.record()
        for _ in range(args.num_iter):
            _ = model(sent1, sent2)
        end.record()
    torch.cuda.synchronize()

    # Print result
    elapsed_time = start.elapsed_time(end) / args.num_iter
    print(f"Elapsed time ({args.batch_size = }) = {elapsed_time}")
    print(f"Number of sentence per 1 second = {args.batch_size / elapsed_time * 1000}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main(parser.parse_args())
