import argparse
import json
import random

import matplotlib.pyplot as plt
import torch
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sentsim.config import ModelArguments
from sentsim.data.sts import load_sts13
from sentsim.models.models import create_contrastive_learning
from transformers import AutoTokenizer


def plot_heatmap(s1, s2, data, score, fpath):
    l1, l2 = len(s1), len(s2)

    fig, ax = plt.subplots(figsize=(int(0.5 * len(s2)), int(0.4 * len(s1))), facecolor="white")

    plt.gca().invert_yaxis()
    ax.set_title(f"score: {score:.3f}")
    ax.set_yticks([y + 0.5 for y in range(l1)])
    ax.set_yticklabels(s1)
    ax.set_xticks([x + 0.5 for x in range(l2)])
    ax.set_xticklabels(s2, rotation=90)
    ax.xaxis.set_ticks_position("top")
    im = ax.pcolormesh(data, edgecolors="k", linewidths=1, cmap=plt.get_cmap("Blues"))

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    plt.tight_layout()
    plt.savefig(fpath, dpi=400)


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--sent1-path", type=str)
parser.add_argument("--sent2-path", type=str)
parser.add_argument("--model-args-path", type=str)
parser.add_argument("--ckpt-path", type=str)


def main():
    args = parser.parse_args()
    sts13 = load_sts13("/nas/home/sh0416/data/STS/STS13-en-test/")

    with open(args.model_args_path) as f:
        model_args = ModelArguments(**json.load(f))
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    module = create_contrastive_learning(model_args)
    module.load_state_dict(torch.load(args.ckpt_path))
    model = module.model
    model.eval()
    step = 0
    with torch.no_grad():
        examples = [x for examples in sts13.values() for x in examples]
        random.seed(0)
        examples = random.sample(examples, k=60) + sorted(examples, key=lambda x: x[1])[-30:]
        examples = sorted(examples, key=lambda x: x[1])
        # examples = examples[:30] + examples[-30:]
        for (s1, s2), score in examples:
            t1 = tokenizer.convert_ids_to_tokens(tokenizer(s1)["input_ids"])
            t2 = tokenizer.convert_ids_to_tokens(tokenizer(s2)["input_ids"])
            if len(t1) > 12 or len(t2) > 12:
                continue
            x1 = tokenizer(s1, padding=True, return_tensors="pt")
            x2 = tokenizer(s2, padding=True, return_tensors="pt")
            heatmap = model.compute_heatmap(x1, x2)[0].numpy()
            plot_heatmap(t1, t2, heatmap, score, f"case_{step}_avg.png")
            step += 1


if __name__ == "__main__":
    main()
