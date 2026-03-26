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

    # ===== CSV HEADER =====
    if f:
        f.write(
            "user_id,Answer_item,rank,"
            "top1,top2,top3,top4,top5,top6,top7,top8,top9,top10,hit\n"
        )

    with torch.no_grad():
        users_emb_all, items_emb_all = model()

        for u in users:
            pos_i = test_item[u]
            seen = set(train_items[u])
            seen.add(pos_i)

            # negative sampling (기존과 동일)
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

            # ===== CSV logging =====
            if f:
                top10 = ranked_items[:10].tolist()

                if pos_i in top10:
                    rank = top10.index(pos_i) + 1
                    hit_str = "HIT"
                else:
                    rank = -1
                    hit_str = "MISS"

                row = (
                    f"{u},{pos_i},{rank},"
                    + ",".join(map(str, top10))
                    + f",{hit_str}\n"
                )
                f.write(row)

            # ===== Hit@K 계산 (기존과 완전 동일) =====
            for k in k_list:
                metrics[k].append(
                    1.0 if pos_i in ranked_items[:k] else 0.0
                )

    if f:
        f.close()

    return {k: float(np.mean(metrics[k])) for k in k_list}
