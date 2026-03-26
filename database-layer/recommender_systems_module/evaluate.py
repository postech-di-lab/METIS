import torch
import numpy as np
import random
import sys

from dataload_loo import CiteULikeLoader
from model import LightGCN
from loo import evaluate_loo
# from loo_logging import evaluate_loo_with_logging


MODEL_PATH = "/home/kai0920/Verification/Accuracy-2025/model.pth"
DATASET_PATH = "/home/kai0920/Verification/Accuracy-2025/citeulike/citeulike-a/users.dat"

SEED_LIST = list(range(1, 11))
TOP_K_LIST = [10]


class Logger(object):
    def __init__(self):
        self.terminal = sys.stdout
        self.log = open("/home/kai0920/Verification/Accuracy-2025/LIGHTGCN_TEST_SEEDS_LOG.TXT", "w", encoding="utf-8")

    def write(self, msg):
        self.terminal.write(msg)
        self.log.write(msg)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def main():
    sys.stdout = Logger()
    print("사용자 맞춤 논문 추천 시스템에 대한 정확도 성능 평가\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 데이터 로드
    dataset = CiteULikeLoader(DATASET_PATH, device)
    graph = dataset.get_sparse_graph()

    # 모델 로드
    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    cfg = checkpoint.get("config", {
        "emb_dim": 512,
        "lr": 0.001,
        "reg": 0.001,
        "layers": 5,
        "batch_size": 2048,
    })

    model = LightGCN(
        dataset.n_users,
        dataset.n_items,
        cfg,
        graph
    ).to(device)

    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    for seed in SEED_LIST:
        set_seed(seed)

        log_path = f"/home/kai0920/Verification/Accuracy-2025/TEST_LOG_SEED_{seed}.txt"

        hit10 = evaluate_loo(
            model,
            dataset,
            device,
            k_list=[10],
            n_neg=99,
            log_file=log_path
        )[10]

        print(f"[Seed {seed}] Hit@10 Accuracy={hit10 * 100:.3f}%")
        print(f"[Seed {seed}] log 저장 위치: {log_path}\n")

if __name__ == "__main__":
    main()
