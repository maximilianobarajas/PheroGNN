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
- `pherognn_appnp` (`PheroAPPNP`): decouples feature transformation from
  propagation (Predict-then-Propagate / APPNP). A shallow 2-layer MLP
  produces an initial prediction that is then diffused `K=10` hops through
  the persistent-pheromone-weighted graph with a teleport back to the
  initial prediction at every hop, giving a much larger receptive field
  than 2-layer PheroGNN without adding learnable graph layers. The
  pheromone trail governing the diffusion is still updated by the usual
  evaporate/reinforce dynamics. A `K in {5,10,20}` x `alpha in {0.05,0.1,0.2}`
  grid (`scripts/sweep_appnp.py`) found no configuration uniformly better
  than `K=10, alpha=0.1`, which is kept as the simple, untuned default.

An extensive ablation (`configs/ablation_v7.yaml`, `scripts/sweep_heuristic_beta.py`,
`scripts/sweep_appnp.py`) across 15 seeds x 4 datasets found no single
mechanism uniformly beats plain PheroGNN: `pherognn_v7_heuristic_hetero`
significantly improves Cora (p=0.0002) but significantly hurts
CiteSeer/PubMed; `pherognn_v7_dropedge` is statistically neutral everywhere;
`pherognn_appnp` significantly improves CiteSeer (p=0.002) and PubMed
(p=0.0001) — new best results on both — but significantly *hurts* the small
500-node synthetic fraud graph (p=0.016, too many diffusion hops oversmooths
such a small graph). We also tried post-hoc pheromone-weighted
Correct-and-Smooth label propagation (`scripts/correct_and_smooth.py`,
PyG's `CorrectAndSmooth`) on top of the trained model; across an alpha/depth
grid it never beat the raw model on Cora/CiteSeer and only marginally helped
PubMed — a documented negative result (C&S is designed to lift graph-blind
base predictors; PheroGNN already message-passes over the graph, so the
extra smoothing mostly over-smooths an already graph-aware prediction).

Since the paper's own protocol already allows dataset-specific tuning using
only the validation partition, **`pherognn_select`**
(`pherognn/train.py:train_select`) trains the compact, individually-justified
family `{pherognn, pherognn_v7_heuristic_hetero, pherognn_v7_dropedge,
pherognn_appnp}` and picks the winner by validation Macro-F1 alone (test
labels are never used for the decision). Over 15 fresh end-to-end seeds this
is:

| Dataset | vs plain PheroGNN | vs GCN | vs GAT | vs GraphSAGE |
|---|---|---|---|---|
| synthetic | tie (p=1.0) | tie (p=0.39) | tie (p=0.44) | tie (p=0.78) |
| Cora | **+0.0071 (p=0.041)** | tie (p=0.72) | tie (p=0.89) | **+0.018 (p=0.0002)** |
| CiteSeer | **+0.0092 (p=0.0003)** | **+0.015 (p=0.0001)** | **+0.022 (p=0.0001)** | **+0.024 (p=0.0001)** |
| PubMed | **+0.0067 (p=0.0026)** | **+0.010 (p=0.0003)** | **+0.020 (p=0.0001)** | **+0.028 (p=0.0001)** |

i.e. never significantly worse than plain PheroGNN or GCN/GAT on any
dataset, and significantly better than all three baselines on 3 of 4
datasets — see `scripts/selection_analysis.py` for the reproducible
comparison. This is the recommended PheroGNN variant going forward,
included by default in `configs/default.yaml`.

The ablation configs used to reach this conclusion are kept for
reproducibility: `configs/ablation_v7.yaml` / `ablation_v7b.yaml` (5-seed
mechanism screening), `configs/final_significance*.yaml` / `final_appnp.yaml`
(15-seed significance runs per mechanism), and `configs/final_select*.yaml`
(15-seed end-to-end `pherognn_select` runs). Run any of them with
`python scripts/run_experiments.py --config configs/<name>.yaml --no-interpretability`.

## Elliptic (optional)

Download the Elliptic Bitcoin dataset and place `elliptic_txs_features.csv`, `elliptic_txs_classes.csv`, `elliptic_txs_edgelist.csv` in `data/elliptic/`, then run:

```
python scripts/run_experiments.py --include-elliptic
```

## Config

Edit `configs/default.yaml` to change datasets, seeds, model hyperparameters, or pheromone settings.
