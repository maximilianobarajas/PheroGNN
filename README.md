# PheroGNN

Graph neural networks (GCN, GAT, GraphSAGE, PheroGNN, PheroGNN-Select) benchmarked on Cora, CiteSeer, PubMed, a synthetic fraud graph, and optionally Elliptic.

## Setup

```
pip install -r requirements.txt
```

## Run

```
python scripts/run_experiments.py
python scripts/analyze_results.py
python scripts/generate_figures.py
```

Results land in `results/`: per-run histories, trained models, pheromone edge weights, statistical comparisons, LaTeX tables, and figures.

## PheroGNN v7 mechanisms and PheroGNN-Select

`pherognn/models.py` adds several pheromone-routing mechanisms inspired by
recent ACO/GNN literature, each toggleable independently on `PheroGNNv7`
(see `PHEROGNN_V7_VARIANTS`):

- `pherognn_v7_heuristic`: the classical ACO transition rule
  `p_ij ~ tau_ij^alpha * eta_ij^beta`, where `eta` is a learned GAT-style
  heuristic jointly guiding routing alongside the persistent pheromone trail
  (graph-transformer/neural-guided-ACO style).
- `pherognn_v7_dual` / `pherognn_v7_hetero`: separate positive/negative
  ("repellent") pheromone trails, optionally reinforced by an unsupervised
  heterophily-disagreement signal on all edges (graph-rewiring /
  heterophilic-edge-pruning literature).
- `pherognn_v7_gate`: a per-node learned gate mixing 1-hop and 2-hop
  representations (node-wise prioritized propagation).
- `pherognn_v7_dropedge`: layer-dependent structure-aware DropEdge
  regularization.
- `pherognn_v7_heuristic_adaptive`: the heuristic weight `beta` is learned
  instead of fixed, so the model can suppress it on label-scarce graphs.

An extensive ablation (`configs/ablation_v7.yaml`, `scripts/sweep_heuristic_beta.py`)
across 15 seeds x 4 datasets found no single mechanism uniformly beats plain
PheroGNN: `pherognn_v7_heuristic_hetero` significantly improves Cora
(p=0.0002) but significantly hurts CiteSeer/PubMed, while
`pherognn_v7_dropedge` is statistically neutral everywhere. Since the paper's
own protocol already allows dataset-specific tuning using only the
validation partition, **`pherognn_select`** (`pherognn/train.py:train_select`)
trains the small, well-justified family `{pherognn, pherognn_v7_heuristic_hetero,
pherognn_v7_dropedge}` and picks the winner by validation Macro-F1 alone
(test labels are never used for the decision). Over 15 seeds this is never
significantly worse than plain PheroGNN on any of the four benchmarks
(p >= 0.075 everywhere) while significantly beating GAT and GraphSAGE on
Cora/CiteSeer/PubMed and GCN on CiteSeer — see `scripts/selection_analysis.py`
for the reproducible comparison. This is the recommended PheroGNN variant
going forward, included by default in `configs/default.yaml`.

The ablation configs used to reach this conclusion are kept for
reproducibility: `configs/ablation_v7.yaml` / `ablation_v7b.yaml` (5-seed
mechanism screening), `configs/final_significance*.yaml` (15-seed
significance runs per mechanism), and `configs/final_select.yaml` (15-seed
end-to-end `pherognn_select` run). Run any of them with
`python scripts/run_experiments.py --config configs/<name>.yaml --no-interpretability`.

## Elliptic (optional)

Download the Elliptic Bitcoin dataset and place `elliptic_txs_features.csv`, `elliptic_txs_classes.csv`, `elliptic_txs_edgelist.csv` in `data/elliptic/`, then run:

```
python scripts/run_experiments.py --include-elliptic
```

## Config

Edit `configs/default.yaml` to change datasets, seeds, model hyperparameters, or pheromone settings.
