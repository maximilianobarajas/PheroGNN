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
grid = [(K, a) for K in [5, 10, 20] for a in [0.05, 0.1, 0.2]]

results = {}
for ds in datasets:
    for seed in seeds:
        set_seed(seed)
        data = load_dataset(ds, cfg, seed).to(device)
        for K, a in grid:
            set_seed(seed)
            appnp_cfg = dict(cfg.get("pheromone_appnp", {}))
            appnp_cfg["K"] = K
            appnp_cfg["appnp_alpha"] = a
            cfg["pheromone_appnp"] = appnp_cfg
            model = build_model("pherognn_appnp", data, cfg).to(device)
            _, _, final = train_one(model, data, cfg)
            results.setdefault((ds, K, a), []).append(final["macro_f1"])
        print(f"done {ds} seed {seed}", flush=True)

print("\n==== summary (mean macro_f1 over seeds) ====")
for ds in datasets:
    print(f"-- {ds} --")
    for K, a in sorted(grid, key=lambda ka: -sum(results[(ds, *ka)]) / len(results[(ds, *ka)])):
        vals = results[(ds, K, a)]
        mean = sum(vals) / len(vals)
        print(f"  K={K} alpha={a}: {mean:.4f}  ({[round(v,4) for v in vals]})")
