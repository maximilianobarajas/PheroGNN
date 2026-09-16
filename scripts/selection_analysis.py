"""Validation-only model selection across the PheroGNN mechanism family.

For each (dataset, seed), every candidate pherognn-family variant already
reports a val_macro_f1 trajectory in its history CSV. We pick, per (dataset,
seed), whichever candidate has the highest val_macro_f1 at its own selected
best_epoch (never touching test labels for the decision), and report the
corresponding test macro_f1. This mirrors the existing single-run early
stopping protocol (selection_metric on validation) extended across the small
model family instead of just across epochs.
"""
from pathlib import Path
import pandas as pd

CANDIDATES = [
    "pherognn", "pherognn_v7_heuristic", "pherognn_v7_dropedge",
    "pherognn_v7_heuristic_dropedge", "pherognn_v7_heuristic_hetero",
    "pherognn_v7_dual", "pherognn_v7_heuristic_adaptive",
]
RESULT_DIRS = ["results_final", "results_final2", "results_final3"]
DATASETS = ["synthetic", "cora", "citeseer", "pubmed"]


def load_val_test(dataset, model, seed):
    for root in RESULT_DIRS:
        p = Path(root) / "histories" / f"{dataset}__{model}__seed{seed}.csv"
        if p.exists():
            h = pd.read_csv(p)
            best_row = h.loc[h["val_macro_f1"].idxmax()]
            return float(best_row["val_macro_f1"]), float(best_row["test_macro_f1"])
    return None


def main():
    rows = []
    for dataset in DATASETS:
        for seed in range(15):
            per_model = {}
            for model in CANDIDATES:
                r = load_val_test(dataset, model, seed)
                if r is not None:
                    per_model[model] = r
            if not per_model:
                continue
            chosen_model = max(per_model, key=lambda m: per_model[m][0])
            chosen_val, chosen_test = per_model[chosen_model]
            baseline_test = per_model.get("pherognn", (None, None))[1]
            rows.append({
                "dataset": dataset, "seed": seed, "chosen_model": chosen_model,
                "chosen_val_macro_f1": chosen_val, "chosen_test_macro_f1": chosen_test,
                "pherognn_v6_test_macro_f1": baseline_test,
            })
    df = pd.DataFrame(rows)
    df.to_csv("results_final/selection_analysis.csv", index=False)

    print("==== Per-dataset: selected-by-validation vs fixed PheroGNN v6 ====")
    for dataset, d in df.groupby("dataset"):
        sel_mean = d["chosen_test_macro_f1"].mean()
        v6_mean = d["pherognn_v6_test_macro_f1"].mean()
        picks = d["chosen_model"].value_counts().to_dict()
        print(f"{dataset}: selected_mean={sel_mean:.4f} v6_mean={v6_mean:.4f} "
              f"diff={sel_mean - v6_mean:+.4f} picks={picks}")


if __name__ == "__main__":
    main()
