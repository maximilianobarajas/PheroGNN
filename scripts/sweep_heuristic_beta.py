from pathlib import Path
import sys
import yaml
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pherognn.data import load_dataset, set_seed
from pherognn.models import build_model
from pherognn.train import train_one

cfg = yaml.safe_load(open("configs/default.yaml"))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

datasets = ["synthetic", "cora", "citeseer", "pubmed"]
seeds = [0, 1, 2]
grid = [(a, b) for a in [0.5, 1.0, 2.0] for b in [0.5, 1.0, 2.0]]

results = {}
for ds in datasets:
    for seed in seeds:
        set_seed(seed)
        data = load_dataset(ds, cfg, seed).to(device)
        for alpha, beta in grid:
            set_seed(seed)
            v7 = dict(cfg.get("pheromone_v7", {}))
            v7["alpha"] = alpha
            v7["beta"] = beta
            cfg["pheromone_v7"] = v7
            model = build_model("pherognn_v7_heuristic", data, cfg).to(device)
            _, _, final = train_one(model, data, cfg)
            key = (ds, alpha, beta)
            results.setdefault(key, []).append(final["macro_f1"])
        print(f"done {ds} seed {seed}", flush=True)

print("\n==== summary (mean macro_f1 over seeds) ====")
for ds in datasets:
    print(f"-- {ds} --")
    for alpha, beta in grid:
        vals = results[(ds, alpha, beta)]
        mean = sum(vals) / len(vals)
        print(f"  alpha={alpha} beta={beta}: {mean:.4f}  ({[round(v,4) for v in vals]})")
