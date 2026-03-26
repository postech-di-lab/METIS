import torch
import numpy as np


def evaluate_loo(
    model,
    dataset,
    device,
    k_list=None,
    n_neg=99,
    log_file=None
):
    if k_list is None:
        k_list = [10]

    model.eval()
    test_item = dataset.test_item
    train_items = dataset.train_items

    users = list(test_item.keys())
    metrics = {k: [] for k in k_list}

    f = open(log_file, "w", encoding="utf-8") if log_file else None

    with torch.no_grad():
        users_emb_all, items_emb_all = model()

        for u in users:
            pos_i = test_item[u]
            seen = set(train_items[u])
            seen.add(pos_i)

            # ===== 기존 negative sampling (절대 변경 ❌) =====
            negs = []
            while len(negs) < n_neg:
                neg = np.random.randint(0, dataset.n_items)
                if neg not in seen:
                    negs.append(neg)
            negs = np.array(negs, dtype=np.int64)

            candidates = np.concatenate([[pos_i], negs])

            u_t = torch.LongTensor([u]).to(device)
            c_t = torch.LongTensor(candidates).to(device)

            u_emb = users_emb_all[u_t]
            c_emb = items_emb_all[c_t]

            scores = torch.matmul(u_emb, c_emb.t()).view(-1)
            _, rank_idx = torch.sort(scores, descending=True)

            ranked_items = candidates[rank_idx.cpu().numpy()]

            # ===== logging (Top-10) =====
            if f is not None:
                top10 = ranked_items[:10].tolist()
                result = "HIT" if pos_i in top10 else "MISS"

                f.write(
                    f"[User {u} | "
                    f"Answer Item {pos_i} | "
                    f"Top10 Pred {top10} | "
                    f"{result}]\n"
                )

            # ===== Hit@K 계산 (기존과 동일) =====
            for k in k_list:
                hit = 1.0 if pos_i in ranked_items[:k] else 0.0
                metrics[k].append(hit)

    if f:
        f.close()

    return {k: float(np.mean(metrics[k])) for k in k_list}
