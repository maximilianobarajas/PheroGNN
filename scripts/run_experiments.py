from pathlib import Path
import argparse
import sys
import yaml
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pherognn.data import load_dataset, set_seed
from pherognn.models import build_model
from pherognn.train import train_one, train_select
from pherognn.interpretability import edge_table, node_scores, centrality_analysis


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--include-elliptic", action="store_true")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--no-interpretability", action="store_true",
                     help="Skip edge/centrality interpretability exports (useful for fast ablation sweeps).")
    ap.add_argument("--models", default=None,
                     help="Comma-separated override of experiment.models from the config.")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else ("cpu" if args.device == "auto" else args.device)
    )
    out = Path(cfg["output"]["directory"])
    for d in ["raw", "histories", "pheromones", "interpretability", "models"]:
        (out / d).mkdir(parents=True, exist_ok=True)

    datasets = list(cfg["experiment"]["datasets"])
    if args.include_elliptic and "elliptic" not in datasets:
        datasets.append("elliptic")
    models = args.models.split(",") if args.models else cfg["experiment"]["models"]
    rows = []

    for dataset_name in datasets:
        for seed in cfg["experiment"]["seeds"]:
            set_seed(seed)
            data = load_dataset(dataset_name, cfg, seed).to(device)
            for model_name in models:
                set_seed(seed)
                tag = f"{dataset_name}__{model_name}__seed{seed}"
                print(f"Running {tag} on {device}", flush=True)
                if model_name == "pherognn_select":
                    model, history, result = train_select(build_model, data, cfg)
                else:
                    model = build_model(model_name, data, cfg).to(device)
                    model, history, result = train_one(model, data, cfg)
                history.to_csv(out / "histories" / f"{tag}.csv", index=False)
                torch.save(model.state_dict(), out / "models" / f"{tag}.pt")
                rows.append({"dataset": dataset_name, "model": model_name, "seed": seed, **result})
                pd.DataFrame(rows).to_csv(out / "raw" / "all_runs.csv", index=False)

                if (model_name.startswith("pherognn") and not args.no_interpretability
                        and hasattr(model, "effective_tau")):
                    edge_table(model, data).to_csv(out / "pheromones" / f"{tag}_edges.csv", index=False)
                    ns, corr = centrality_analysis(model, data)
                    ns.to_csv(out / "interpretability" / f"{tag}_nodes.csv", index=False)
                    corr.to_csv(out / "interpretability" / f"{tag}_correlations.csv", index=False)

    print(f"Completed. Results written to {out.resolve()}")


if __name__ == "__main__":
    main()
