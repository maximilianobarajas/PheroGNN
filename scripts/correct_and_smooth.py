"""Post-hoc Correct & Smooth (Huang et al., ICLR 2021) on top of a trained
PheroGNN base model, using the model's learned pheromone as the label
propagation edge weight instead of plain (uniformly-normalized) adjacency.

Tests whether pheromone-weighted label propagation gives a genuine additional
boost over both the raw base model and plain (unweighted) Correct & Smooth.
"""
from pathlib import Path
import sys
import yaml
import torch
from torch_geometric.nn.models import CorrectAndSmooth

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pherognn.data import load_dataset, set_seed
from pherognn.models import build_model
from pherognn.train import train_one, metrics

cfg = yaml.safe_load(open("configs/default.yaml"))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATASETS = ["synthetic", "cora", "citeseer", "pubmed"]
SEEDS = list(range(15))
BASE_MODEL = "pherognn"

CS_KW = dict(num_correction_layers=50, correction_alpha=0.8,
             num_smoothing_layers=50, smoothing_alpha=0.8, autoscale=True)

rows = []
for ds in DATASETS:
    for seed in SEEDS:
        set_seed(seed)
        data = load_dataset(ds, cfg, seed).to(device)
        set_seed(seed)
        model = build_model(BASE_MODEL, data, cfg).to(device)
        model, _, final = train_one(model, data, cfg)
        model.eval()
        with torch.no_grad():
            logits = model(data.x, data.edge_index)
        y_soft = torch.softmax(logits, dim=-1)
        y_true_train = data.y[data.train_mask]

        raw_f1 = final["macro_f1"]

        cs_plain = CorrectAndSmooth(**CS_KW)
        y_c = cs_plain.correct(y_soft, y_true_train, data.train_mask, data.edge_index)
        y_cs_plain = cs_plain.smooth(y_c, y_true_train, data.train_mask, data.edge_index)
        m_plain = metrics(y_cs_plain, data.y, data.test_mask, data.num_classes)

        tau = model.effective_tau().detach()
        cs_phero = CorrectAndSmooth(**CS_KW)
        y_c2 = cs_phero.correct(y_soft, y_true_train, data.train_mask, data.edge_index, edge_weight=tau)
        y_cs_phero = cs_phero.smooth(y_c2, y_true_train, data.train_mask, data.edge_index, edge_weight=tau)
        m_phero = metrics(y_cs_phero, data.y, data.test_mask, data.num_classes)

        rows.append(dict(dataset=ds, seed=seed, raw_macro_f1=raw_f1,
                          cs_plain_macro_f1=m_plain["macro_f1"],
                          cs_phero_macro_f1=m_phero["macro_f1"]))
        print(f"{ds} seed{seed}: raw={raw_f1:.4f} cs_plain={m_plain['macro_f1']:.4f} "
              f"cs_phero={m_phero['macro_f1']:.4f}", flush=True)

import pandas as pd
df = pd.DataFrame(rows)
df.to_csv("C:/tmp/correct_and_smooth_results.csv", index=False)

print("\n==== summary (mean over seeds) ====")
for ds, d in df.groupby("dataset"):
    print(f"{ds}: raw={d.raw_macro_f1.mean():.4f} cs_plain={d.cs_plain_macro_f1.mean():.4f} "
          f"cs_phero={d.cs_phero_macro_f1.mean():.4f}")
