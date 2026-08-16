from __future__ import annotations
from pathlib import Path
from typing import Optional
import random
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.utils import to_undirected, coalesce


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def stratified_masks(y, seed, train_ratio=.6, val_ratio=.2, ignore_label: Optional[int] = None):
    rng = np.random.default_rng(seed)
    n = y.numel()
    train = torch.zeros(n, dtype=torch.bool)
    val = torch.zeros(n, dtype=torch.bool)
    test = torch.zeros(n, dtype=torch.bool)
    y_cpu = y.cpu().numpy()
    for c in np.unique(y_cpu):
        if ignore_label is not None and int(c) == ignore_label:
            continue
        idx = np.where(y_cpu == c)[0]
        rng.shuffle(idx)
        nt = int(len(idx) * train_ratio)
        nv = int(len(idx) * val_ratio)
        train[idx[:nt]] = True
        val[idx[nt:nt + nv]] = True
        test[idx[nt + nv:]] = True
    return train, val, test


def synthetic_fraud(seed=0, num_legit=420, num_fraud=80, num_features=16) -> Data:
    set_seed(seed)
    rng = np.random.default_rng(seed)
    n = num_legit + num_fraud
    x0 = rng.normal(0, 1, (num_legit, num_features))
    x1 = rng.normal(.35, 1.15, (num_fraud, num_features))
    x1[:, :min(4, num_features)] += .65
    x = torch.tensor(np.vstack([x0, x1]), dtype=torch.float)
    y = torch.zeros(n, dtype=torch.long)
    y[num_legit:] = 1

    edges = []
    communities = np.array_split(np.arange(num_legit), 4)
    for com in communities:
        for u in com:
            candidates = rng.choice(com, size=min(5, len(com) - 1), replace=False)
            edges.extend((int(u), int(v)) for v in candidates if u != v)
    fraud_nodes = np.arange(num_legit, n)
    for k, u in enumerate(fraud_nodes):
        edges.append((int(u), int(fraud_nodes[(k + 1) % num_fraud])))
        edges.append((int(u), int(fraud_nodes[(k + 3) % num_fraud])))
    for _ in range(120):
        u, v = rng.choice(fraud_nodes, 2, replace=False)
        edges.append((int(u), int(v)))
    for _ in range(85):
        u = int(rng.choice(fraud_nodes))
        v = int(rng.integers(0, num_legit))
        edges.append((u, v))

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    edge_index = coalesce(to_undirected(edge_index), num_nodes=n)
    train, val, test = stratified_masks(y, seed)
    data = Data(x=x, y=y, edge_index=edge_index, train_mask=train, val_mask=val, test_mask=test)
    data.dataset_name = "synthetic"
    data.num_classes = 2
    return data


def planetoid(name, root="data/planetoid", seed=0, use_public_split=True) -> Data:
    ds = Planetoid(root=root, name=name, transform=T.NormalizeFeatures())
    data = ds[0]
    if not use_public_split:
        data.train_mask, data.val_mask, data.test_mask = stratified_masks(data.y, seed)
    data.dataset_name = name.lower()
    data.num_classes = ds.num_classes
    return data


def elliptic(directory, seed=0, make_undirected=True, temporal_split=True) -> Data:
    directory = Path(directory)
    fpath = directory / "elliptic_txs_features.csv"
    cpath = directory / "elliptic_txs_classes.csv"
    epath = directory / "elliptic_txs_edgelist.csv"
    missing = [str(p) for p in (fpath, cpath, epath) if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing Elliptic files:\n" + "\n".join(missing))

    features = pd.read_csv(fpath, header=None)
    classes = pd.read_csv(cpath)
    edges = pd.read_csv(epath)

    ids = features.iloc[:, 0].astype(str).tolist()
    id_to_idx = {v: i for i, v in enumerate(ids)}
    timestep = features.iloc[:, 1].to_numpy(dtype=int)
    x_np = features.iloc[:, 1:].to_numpy(dtype=np.float32)
    x_np = (x_np - x_np.mean(0, keepdims=True)) / (x_np.std(0, keepdims=True) + 1e-8)

    y_np = np.full(len(ids), -1, dtype=np.int64)
    label_map = {"2": 0, "1": 1, "unknown": -1}
    for _, row in classes.iterrows():
        k = str(row["txId"])
        if k in id_to_idx:
            y_np[id_to_idx[k]] = label_map.get(str(row["class"]), -1)

    src, dst = [], []
    for s, d in zip(edges["txId1"].astype(str), edges["txId2"].astype(str)):
        if s in id_to_idx and d in id_to_idx:
            src.append(id_to_idx[s]); dst.append(id_to_idx[d])
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    if make_undirected:
        edge_index = to_undirected(edge_index)
    edge_index = coalesce(edge_index, num_nodes=len(ids))

    y = torch.tensor(y_np, dtype=torch.long)
    train = torch.zeros(len(ids), dtype=torch.bool)
    val = torch.zeros(len(ids), dtype=torch.bool)
    test = torch.zeros(len(ids), dtype=torch.bool)

    if temporal_split:
        train |= torch.tensor((timestep <= 34) & (y_np >= 0))
        val |= torch.tensor((timestep >= 35) & (timestep <= 41) & (y_np >= 0))
        test |= torch.tensor((timestep >= 42) & (y_np >= 0))
    else:
        train, val, test = stratified_masks(y, seed, ignore_label=-1)

    data = Data(x=torch.tensor(x_np), y=y, edge_index=edge_index, train_mask=train, val_mask=val, test_mask=test)
    data.dataset_name = "elliptic"
    data.num_classes = 2
    data.timestep = torch.tensor(timestep)
    return data


def load_dataset(name: str, cfg: dict, seed: int) -> Data:
    name = name.lower()
    if name == "synthetic":
        c = cfg["synthetic"]
        return synthetic_fraud(seed, c["num_legit"], c["num_fraud"], c["num_features"])
    if name in {"cora", "citeseer", "pubmed"}:
        c = cfg["planetoid"]
        canonical = {"cora": "Cora", "citeseer": "CiteSeer", "pubmed": "PubMed"}[name]
        return planetoid(canonical, c["root"], seed, c["use_public_split"])
    if name == "elliptic":
        c = cfg["elliptic"]
        return elliptic(c["directory"], seed, c["make_undirected"], c["temporal_split"])
    raise ValueError(f"Unknown dataset: {name}")
