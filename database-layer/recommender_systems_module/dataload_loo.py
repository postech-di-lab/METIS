# # dataload_loo.py
# import os
# import numpy as np
# import torch
# import scipy.sparse as sp
# from collections import defaultdict

# class CiteULikeLoader:
#     def __init__(self, path, device):
#         self.path = path
#         self.device = device

#         self.train_items = defaultdict(list)  # user -> [items]
#         self.test_item = {}                   # user -> single item
#         self.n_users = 0
#         self.n_items = 0
#         self.Graph = None
#         self.train_data = None

#         self._load_and_split()

#     def _load_and_split(self):
#         if not os.path.exists(self.path):
#             raise FileNotFoundError(self.path)

#         with open(self.path, "r") as f:
#             lines = f.readlines()

#         self.n_users = len(lines)
#         all_items = set()
#         user_interactions = []

#         for uid, line in enumerate(lines):
#             line = line.strip()
#             if not line:
#                 continue
#             items = list(map(int, line.split()))
#             if items:
#                 user_interactions.append((uid, items))
#                 all_items.update(items)

#         self.n_items = max(all_items) + 1 if all_items else 0
#         print(f"[Data] Users: {self.n_users}, Items: {self.n_items}")

#         train_pairs = []
#         for uid, items in user_interactions:
#             if len(items) == 1:
#                 iid = items[0]
#                 self.train_items[uid].append(iid)
#                 train_pairs.append([uid, iid])
#                 continue

#             items = np.array(items, dtype=np.int64)
#             np.random.shuffle(items)
#             test_i = items[-1]
#             train_part = items[:-1]

#             self.test_item[uid] = int(test_i)
#             for iid in train_part:
#                 iid = int(iid)
#                 self.train_items[uid].append(iid)
#                 train_pairs.append([uid, iid])

#         self.train_data = np.array(train_pairs, dtype=np.int64)
#         print(f"[Data] Train interactions: {len(self.train_data)}, Test users: {len(self.test_item)}")

#     def get_sparse_graph(self):
#         if self.Graph is not None:
#             return self.Graph

#         print("[Data] Building user-item graph (LOO)...")
#         users = self.train_data[:, 0]
#         items = self.train_data[:, 1]

#         row_idx = np.concatenate([users, items + self.n_users])
#         col_idx = np.concatenate([items + self.n_users, users])
#         data_vals = np.ones(len(row_idx), dtype=np.float32)

#         n_nodes = self.n_users + self.n_items
#         adj = sp.coo_matrix((data_vals, (row_idx, col_idx)), shape=(n_nodes, n_nodes))

#         rowsum = np.array(adj.sum(1))
#         d_inv_sqrt = np.power(rowsum, -0.5).flatten()
#         d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
#         d_mat_inv_sqrt = sp.diags(d_inv_sqrt)

#         norm_adj = d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt).tocoo()
#         indices = torch.from_numpy(np.vstack((norm_adj.row, norm_adj.col)).astype(np.int64))
#         values = torch.from_numpy(norm_adj.data.astype(np.float32))
#         shape = torch.Size(norm_adj.shape)

#         self.Graph = torch.sparse_coo_tensor(indices, values, shape).to(self.device)
#         return self.Graph
# dataload_loo.py
import os
import numpy as np
import torch
import scipy.sparse as sp
from collections import defaultdict


class CiteULikeLoader:
    def __init__(self, path, device):
        self.path = path
        self.device = device

        self.train_items = defaultdict(list)  # user -> [items]
        self.valid_item = {}                 # user -> single item (optional)
        self.test_item = {}                  # user -> single item
        self.n_users = 0
        self.n_items = 0
        self.Graph = None
        self.train_data = None              # [u, i] pairs for training
        self.valid_data = None              # [u, i] pairs for validation
        self.test_data = None               # [u, i] pairs for test (for 통계용)

        self._load_and_split()

    def _load_and_split(self):
        if not os.path.exists(self.path):
            raise FileNotFoundError(self.path)

        with open(self.path, "r") as f:
            lines = f.readlines()

        self.n_users = len(lines)
        all_items = set()
        user_interactions = []

        for uid, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
            items = list(map(int, line.split()))
            if items:
                user_interactions.append((uid, items))
                all_items.update(items)

        self.n_items = max(all_items) + 1 if all_items else 0
        # print(f"[Data] Users: {self.n_users}, Items: {self.n_items}")

        train_pairs = []
        valid_pairs = []
        test_pairs = []

        for uid, items in user_interactions:
            items = np.array(items, dtype=np.int64)

            if len(items) == 1:
                # 하나뿐이면 train에만 사용, valid/test 없음
                iid = int(items[0])
                self.train_items[uid].append(iid)
                train_pairs.append([uid, iid])
                continue

            if len(items) == 2:
                # 1개는 train, 1개는 test (valid 없음)
                np.random.shuffle(items)
                train_part = items[:1]
                test_i = items[1]

                self.test_item[uid] = int(test_i)
                test_pairs.append([uid, int(test_i)])

                for iid in train_part:
                    iid = int(iid)
                    self.train_items[uid].append(iid)
                    train_pairs.append([uid, iid])
                continue

            # len(items) >= 3 인 경우: train / valid / test 모두 존재
            np.random.shuffle(items)
            test_i = items[-1]
            valid_i = items[-2]
            train_part = items[:-2]

            self.test_item[uid] = int(test_i)
            self.valid_item[uid] = int(valid_i)

            test_pairs.append([uid, int(test_i)])
            valid_pairs.append([uid, int(valid_i)])

            for iid in train_part:
                iid = int(iid)
                self.train_items[uid].append(iid)
                train_pairs.append([uid, iid])

        self.train_data = np.array(train_pairs, dtype=np.int64)
        self.valid_data = np.array(valid_pairs, dtype=np.int64) if len(valid_pairs) > 0 else np.zeros((0, 2), dtype=np.int64)
        self.test_data = np.array(test_pairs, dtype=np.int64) if len(test_pairs) > 0 else np.zeros((0, 2), dtype=np.int64)

        # print(f"[Data] Train interactions: {len(self.train_data)}, "
        #       f"Valid users: {len(self.valid_item)}, "
        #       f"Test users: {len(self.test_item)}")

    def get_sparse_graph(self):
        if self.Graph is not None:
            return self.Graph

        # print("[Data] Building user-item graph (LOO, train only)...")
        users = self.train_data[:, 0]
        items = self.train_data[:, 1]

        row_idx = np.concatenate([users, items + self.n_users])
        col_idx = np.concatenate([items + self.n_users, users])
        data_vals = np.ones(len(row_idx), dtype=np.float32)

        n_nodes = self.n_users + self.n_items
        adj = sp.coo_matrix((data_vals, (row_idx, col_idx)), shape=(n_nodes, n_nodes))

        rowsum = np.array(adj.sum(1))
        eps = 1e-12
        d_inv_sqrt = np.power(rowsum + eps, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)

        norm_adj = d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt).tocoo()
        indices = torch.from_numpy(np.vstack((norm_adj.row, norm_adj.col)).astype(np.int64))
        values = torch.from_numpy(norm_adj.data.astype(np.float32))
        shape = torch.Size(norm_adj.shape)

        self.Graph = torch.sparse_coo_tensor(indices, values, shape).to(self.device)
        return self.Graph
