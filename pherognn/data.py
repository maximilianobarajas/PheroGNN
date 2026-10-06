from __future__ import annotations
from pathlib import Path
from typing import Optional
import random
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from torch_geometric.data import Data
from torch_geometric.datasets import Amazon, Coauthor, Planetoid, WebKB
import torch_geometric.transforms as T
from torch_geometric.utils import to_undirected, coalesce


def two_hop_edge_index(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """Strict 2-hop edges: pairs reachable in exactly 2 steps, excluding any
    pair that is already a 1-hop edge or a self-loop. Used by PheroH2 to give
    heterophilic graphs ("my neighbor's neighbor is more likely my class"
    -- the H2GCN / monophily observation) an explicit, separate channel
    rather than folding 2-hop information into a 1-hop-only aggregator."""
    src, dst = edge_index.cpu().numpy()
    A = sp.coo_matrix((np.ones(len(src)), (src, dst)), shape=(num_nodes, num_nodes)).tocsr()
    A.data[:] = 1
    A2 = (A @ A).tocsr()
    A2.data[:] = 1
    A2 = A2 - A2.multiply(A)
    A2.setdiag(0)
    A2.eliminate_zeros()
    A2 = A2.tocoo()
    return torch.tensor(np.stack([A2.row, A2.col]), dtype=torch.long)


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


def webkb(name, root="data/webkb", seed=0) -> Data:
    """Small, strongly heterophilic web-page graphs (Texas/Wisconsin/Cornell,
    geom-gcn splits) — a natural stress test for the heterophily-aware
    pheromone mechanisms, which target exactly this regime. Ships with 10
    standard random splits; seed selects one (wrapping past 10) so repeated
    seeds still vary the split like every other dataset here."""
    ds = WebKB(root=root, name=name, transform=T.NormalizeFeatures())
    data = ds[0]
    split = seed % data.train_mask.size(1)
    data.train_mask = data.train_mask[:, split]
    data.val_mask = data.val_mask[:, split]
    data.test_mask = data.test_mask[:, split]
    data.dataset_name = name.lower()
    data.num_classes = ds.num_classes
    return data


def amazon(name, root="data/amazon", seed=0) -> Data:
    """Amazon co-purchase graphs (Photo/Computers) — larger homophilic
    benchmarks with no official split, so we stratify like the synthetic
    fraud graph. Unlike Planetoid's binary bag-of-words, these are sparse
    count features; L1 row-normalization squashes them into a range our
    fixed learning rate cannot escape (GCN macro-F1 0.16 vs 0.92 raw), so
    features are used as-is here."""
    ds = Amazon(root=root, name=name)
    data = ds[0]
    data.train_mask, data.val_mask, data.test_mask = stratified_masks(data.y, seed)
    data.dataset_name = f"amazon_{name.lower()}"
    data.num_classes = ds.num_classes
    return data


def coauthor(name, root="data/coauthor", seed=0) -> Data:
    """Microsoft Academic co-authorship graphs (CS/Physics) — larger
    homophilic benchmarks with no official split, stratified like synthetic.
    Same raw-feature rationale as `amazon()` (row-normalization stalls
    training: GCN macro-F1 0.72 normalized vs 0.92 raw)."""
    ds = Coauthor(root=root, name=name)
    data = ds[0]
    data.train_mask, data.val_mask, data.test_mask = stratified_masks(data.y, seed)
    data.dataset_name = f"coauthor_{name.lower()}"
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
        data = synthetic_fraud(seed, c["num_legit"], c["num_fraud"], c["num_features"])
    elif name in {"cora", "citeseer", "pubmed"}:
        c = cfg["planetoid"]
        canonical = {"cora": "Cora", "citeseer": "CiteSeer", "pubmed": "PubMed"}[name]
        data = planetoid(canonical, c["root"], seed, c["use_public_split"])
    elif name in {"texas", "wisconsin", "cornell"}:
        c = cfg.get("webkb", {"root": "data/webkb"})
        canonical = {"texas": "Texas", "wisconsin": "Wisconsin", "cornell": "Cornell"}[name]
        data = webkb(canonical, c.get("root", "data/webkb"), seed)
    elif name in {"amazon_photo", "amazon_computers"}:
        c = cfg.get("amazon", {"root": "data/amazon"})
        canonical = {"amazon_photo": "Photo", "amazon_computers": "Computers"}[name]
        data = amazon(canonical, c.get("root", "data/amazon"), seed)
    elif name in {"coauthor_cs", "coauthor_physics"}:
        c = cfg.get("coauthor", {"root": "data/coauthor"})
        canonical = {"coauthor_cs": "CS", "coauthor_physics": "Physics"}[name]
        data = coauthor(canonical, c.get("root", "data/coauthor"), seed)
    elif name == "elliptic":
        c = cfg["elliptic"]
        data = elliptic(c["directory"], seed, c["make_undirected"], c["temporal_split"])
    else:
        raise ValueError(f"Unknown dataset: {name}")

    data.edge_index_2hop = two_hop_edge_index(data.edge_index, data.num_nodes)
    return data
