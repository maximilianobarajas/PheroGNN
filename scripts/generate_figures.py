from pathlib import Path
import argparse
import pandas as pd
import matplotlib.pyplot as plt


def save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results/raw/all_runs.csv")
    ap.add_argument("--root", default="results")
    args = ap.parse_args()
    root = Path(args.root)
    out = root / "figures"
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.results)

    for metric in ["macro_f1", "accuracy", "pr_auc", "fraud_f1"]:
        if metric not in df.columns:
            continue
        x = df.groupby(["dataset", "model"])[metric].agg(["mean", "std"]).reset_index()
        for dataset, d in x.groupby("dataset"):
            fig, ax = plt.subplots(figsize=(7, 4.5))
            ax.bar(d["model"], d["mean"], yerr=d["std"], capsize=4)
            ax.set_ylabel(metric.replace("_", " ").title())
            ax.set_title(f"{dataset}: {metric.replace('_', ' ')}")
            ax.tick_params(axis="x", rotation=25)
            save(fig, out / f"{dataset}_{metric}_models.pdf")
            fig, ax = plt.subplots(figsize=(7, 4.5))
            ax.bar(d["model"], d["mean"], yerr=d["std"], capsize=4)
            ax.set_ylabel(metric.replace("_", " ").title())
            ax.set_title(f"{dataset}: {metric.replace('_', ' ')}")
            ax.tick_params(axis="x", rotation=25)
            save(fig, out / f"{dataset}_{metric}_models.png")

    for p in (root / "histories").glob("*__pherognn__seed0.csv"):
        h = pd.read_csv(p)
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.plot(h["epoch"], h["val_macro_f1"], label="Validation")
        ax.plot(h["epoch"], h["test_macro_f1"], label="Test")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Macro-F1")
        ax.set_title(p.stem.split("__")[0] + ": learning curve")
        ax.legend()
        save(fig, out / f"{p.stem}_learning_curve.pdf")

    for p in (root / "pheromones").glob("*__pherognn__seed0_edges.csv"):
        e = pd.read_csv(p)
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.hist(e["pheromone"], bins=30)
        ax.set_xlabel("Final pheromone value")
        ax.set_ylabel("Edges")
        ax.set_title(p.stem.split("__")[0] + ": pheromone distribution")
        save(fig, out / f"{p.stem}_histogram.pdf")

    print(f"Figures written to {out.resolve()}")


if __name__ == "__main__":
    main()
