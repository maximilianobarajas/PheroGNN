from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests

METRICS = ["accuracy", "macro_f1", "balanced_accuracy", "roc_auc", "pr_auc",
           "fraud_precision", "fraud_recall", "fraud_f1", "runtime_seconds"]


def mean_std_table(df):
    cols = [c for c in METRICS if c in df.columns]
    agg = df.groupby(["dataset", "model"])[cols].agg(["mean", "std"])
    agg.columns = [f"{a}_{b}" for a, b in agg.columns]
    return agg.reset_index()


def significance(df, reference="pherognn"):
    rows = []
    for dataset, d in df.groupby("dataset"):
        ref = d[d.model == reference].set_index("seed")
        for model in sorted(set(d.model) - {reference}):
            comp = d[d.model == model].set_index("seed")
            common = ref.index.intersection(comp.index)
            for metric in ["macro_f1", "accuracy", "pr_auc"]:
                if metric not in d.columns or len(common) < 3:
                    continue
                x = ref.loc[common, metric].to_numpy(dtype=float)
                y = comp.loc[common, metric].to_numpy(dtype=float)
                valid = np.isfinite(x) & np.isfinite(y)
                x, y = x[valid], y[valid]
                if len(x) < 3:
                    continue
                if np.allclose(x, y):
                    stat, p = 0.0, 1.0
                else:
                    stat, p = wilcoxon(x, y, alternative="two-sided")
                rows.append({
                    "dataset": dataset, "reference": reference, "comparison": model,
                    "metric": metric, "wilcoxon_W": stat, "p_raw": p,
                    "mean_difference": float(np.mean(x - y)), "n_pairs": len(x),
                })
    out = pd.DataFrame(rows)
    if len(out):
        out["p_holm"] = multipletests(out["p_raw"], method="holm")[1]
    return out


def latex_table(summary, metric="macro_f1"):
    x = summary[["dataset", "model", f"{metric}_mean", f"{metric}_std"]].copy()
    x["value"] = x.apply(lambda r: f"{r[f'{metric}_mean']:.4f} $\\pm$ {r[f'{metric}_std']:.4f}", axis=1)
    piv = x.pivot_table(index="model", columns="dataset", values="value", aggfunc="first")
    return piv.to_latex(escape=False, caption=f"Mean test {metric} over repeated seeds.", label=f"tab:{metric}_results")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results/raw/all_runs.csv")
    ap.add_argument("--output", default="results/tables")
    args = ap.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.results)
    summary = mean_std_table(df)
    summary.to_csv(out / "summary_mean_std.csv", index=False)
    significance(df).to_csv(out / "paired_wilcoxon_holm.csv", index=False)
    for metric in ["macro_f1", "accuracy", "pr_auc", "fraud_f1"]:
        if metric in df.columns:
            (out / f"{metric}_table.tex").write_text(latex_table(summary, metric))
    print(f"Tables written to {out.resolve()}")


if __name__ == "__main__":
    main()
