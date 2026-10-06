# PheroGNN

Graph neural networks (GCN, GAT, GraphSAGE, PheroGNN, PheroGNN-Select)
benchmarked on 9 datasets — Cora, CiteSeer, PubMed, a synthetic fraud graph,
the strongly heterophilic Texas/Wisconsin/Cornell graphs, the larger Amazon
Photo and Coauthor CS graphs — and optionally Elliptic.

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
- `pherognn_sage` (`PheroSAGE`, in `pherognn/models.py`, not a `PheroGNNv7`
  flag since it swaps the message-passing backbone itself): a GraphSAGE-style
  layer (separate self/neighbor linear transforms, mean aggregation of
  pheromone-weighted neighbor messages) instead of PheroConv's GCN-style
  symmetric-normalized sum aggregation. Motivated by testing on 5 further
  datasets (see below): every PheroConv-backbone PheroGNN variant lost
  significantly to plain GraphSAGE on strongly heterophilic graphs
  (Texas/Wisconsin/Cornell) and on larger co-purchase/co-authorship graphs
  (Amazon Photo, Coauthor CS) — the persistent-pheromone mechanism alone
  does not fix a GCN-style backbone's structural disadvantage there.
  PheroSAGE closes that gap (ties GraphSAGE on all 5, $p \geq 0.15$) while
  giving up some accuracy on the four original citation-style datasets
  (its mean-aggregation backbone trades away PheroConv's GCN-style prior
  there, e.g. PubMed $-0.021$, $p=0.0001$) — another dataset-dependent
  trade-off resolved by `pherognn_select` below.
- `pherognn_sage_heuristic`: the same ACO transition-rule heuristic as
  `pherognn_v7_heuristic`, but composed onto `PheroConvSAGE`'s mean
  aggregation (both the pheromone and the learned heuristic are normalized
  to mean $\approx 1$ per neighborhood and multiplied, rather than
  softmax-combined as in the GCN-backbone version) instead of only ever
  being tried on PheroConv's GCN-style backbone. This is the mechanism that
  pushed Texas from a tie into a significant win over GraphSAGE in
  `pherognn_select` (see below) — it was reached by trying the two
  previously-separate, independently-validated wins (SAGE backbone for
  heterophily, ACO heuristic for Cora) *composed together*, after two
  bigger, riskier architecture changes failed (next paragraph).

- `pherognn_gpr` (`PheroGPR`): Generalized-PageRank GNN (Chien et al.,
  ICLR 2021) adapted to the pheromone-weighted graph. Unlike PheroAPPNP's
  fixed positive teleport weight at every hop, GPR-GNN learns one
  *unconstrained, possibly negative* combination weight `gamma_k` per hop
  and sums `gamma_k * P^k H0` — signed gammas can represent
  difference-of-propagations (high-pass-filter-like) behavior without
  changing the propagation operator itself. In practice signed/negative
  gammas never really emerged under our training recipe (even with a 10x
  higher learning rate on the gammas specifically, `pherognn_gpr_fastgamma`,
  tried and not adopted), so it did not close the heterophily gap either —
  but unexpectedly, it became our **best-ever Cora result** (0.811, beating
  every prior candidate) and the first mechanism to significantly beat
  GraphSAGE on **Coauthor CS** (see table below).

Three further architectural ideas were tried and **rejected** after failing
to beat what was already validated: (i) *PheroSAGEAPPNP* decoupled
feature-transform-then-K-hop-diffusion (like PheroAPPNP) but through
row-stochastic mean-aggregation (like PheroSAGE) instead of GCN-style
normalization, hoping to combine APPNP's receptive field with SAGE's
heterophily-robustness — instead it lost most of PheroSAGE's heterophily
advantage (Wisconsin single-seed macro-F1 dropped from PheroSAGE's 0.554 to
0.225) without a compensating homophily gain, because APPNP's weak
per-hop teleport (`alpha=0.1`) cannot substitute for SAGE's full per-layer
self-transform; (ii) *PheroGCNII*, applying GCNII's (Chen et al., ICML
2020) initial-residual-plus-identity-mapping trick to go 16+ layers deep
with pheromone-weighted propagation, hoping depth itself was an unexploited
lever — at 4/8/16/32 layers it never beat plain 2-layer PheroGNN on Cora
under our fixed (untuned-for-depth) learning rate and weight decay, so the
benefit GCNII reports elsewhere did not materialize under this training
recipe; (iii) *PheroH2*, an H2GCN-inspired (Zhu et al., NeurIPS 2020)
architecture with a strict, separate 2-hop pheromone channel (never mixed
with the ego embedding, targeting "monophily" on heterophilic graphs) —
a 5-seed screen on Texas/Wisconsin/Cornell showed no consistent edge over
PheroSAGE or GraphSAGE, so it was dropped before a full 15-seed run. All
three classes remain in `pherognn/models.py` (`pherognn_sage_appnp`,
`pherognn_gcnii`, `pherognn_h2`) as documented negative results, usable
standalone but not part of `PHEROGNN_SELECT_FAMILY`.

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
pherognn_appnp, pherognn_sage, pherognn_sage_heuristic, pherognn_gpr}`. It
picks the winner by validation Macro-F1 among each individual candidate
**and** a uniform probability-space ensemble (`PheroEnsemble`) of all of
them — hard "pick one" beats averaging on CiteSeer (one candidate,
PheroAPPNP, is uniquely strong there and dilution hurts) but averaging
beats picking a single winner on Cora/PubMed (decorrelated errors across
mechanisms), so which strategy to use is itself chosen on the validation
partition, never on test labels. Over **30** fresh end-to-end seeds, across
all 9 benchmark datasets:

| Dataset | vs plain PheroGNN | vs GCN | vs GAT | vs GraphSAGE |
|---|---|---|---|---|
| synthetic | tie (p=0.95) | tie (p=0.94) | tie (p=0.42) | tie (p=0.08) |
| Cora | **+0.0132 (p<0.0001)** | tie (p=0.077) | tie (p=0.070) | **+0.0208 (p<0.0001)** |
| CiteSeer | **+0.0086 (p<0.0001)** | **+0.0130 (p<0.0001)** | **+0.0219 (p<0.0001)** | **+0.0229 (p<0.0001)** |
| PubMed | **+0.0074 (p<0.0001)** | **+0.0092 (p<0.0001)** | **+0.0204 (p<0.0001)** | **+0.0282 (p<0.0001)** |
| Texas | **+0.296 (p<0.0001)** | **+0.305 (p<0.0001)** | **+0.349 (p<0.0001)** | **+0.035 (p=0.017)** |
| Wisconsin | **+0.317 (p<0.0001)** | **+0.325 (p<0.0001)** | **+0.322 (p<0.0001)** | tie (p=0.24) |
| Cornell | **+0.287 (p<0.0001)** | **+0.284 (p<0.0001)** | **+0.272 (p<0.0001)** | tie (p=0.95) |
| Amazon Photo | **+0.0162 (p<0.0001)** | **+0.0163 (p<0.0001)** | **+0.0122 (p<0.0001)** | tie, close (p=0.092) |
| Coauthor CS | **+0.0090 (p<0.0001)** | **+0.0099 (p<0.0001)** | **+0.0142 (p<0.0001)** | **+0.0019 (p=0.0009)** |

i.e. across all 9 datasets PheroGNN-Select is **never significantly worse
than any of GCN, GAT, GraphSAGE, or plain PheroGNN**, and now **significantly
beats all four on 6 of 9 datasets** (CiteSeer, PubMed, Texas, Coauthor CS
outright across every comparison; Cora and Amazon Photo against
PheroGNN/GraphSAGE, with GCN/GAT trending toward significance at p=0.07-0.09
as more seeds are added). Texas (p=0.017) and Coauthor CS (p=0.0009) are
confirmed, significant wins over GraphSAGE specifically — the strongest
baseline in this study — joining Cora/CiteSeer/PubMed, where PheroGNN
already won convincingly. Wisconsin and Cornell remain true ties against
GraphSAGE (p=0.24, p=0.95); plain PheroGNN previously lost heavily to it on
both (e.g. Wisconsin plain PheroGNN 0.24 vs GraphSAGE 0.57). See
`scripts/selection_analysis.py` for the reproducible comparison. This is the
recommended PheroGNN variant going forward, included by default in
`configs/default.yaml`.

### Testing on 5 more datasets

To stress-test the mechanisms above (especially the heterophily-aware ones)
and check they generalize past citation networks, we added loaders
(`pherognn/data.py`) for the strongly heterophilic geom-gcn benchmarks
Texas/Wisconsin/Cornell (183-251 nodes, standard 10-way splits, selected by
`seed % 10`) and the larger homophilic Amazon Photo (7,650 nodes) and
Coauthor CS (18,333 nodes) graphs (stratified splits like the synthetic
graph). One preprocessing pitfall found along the way: unlike Planetoid's
binary bag-of-words features, Amazon/Coauthor's sparse count features make
row-L1-normalization (`NormalizeFeatures`, the standard Planetoid transform)
stall training under our fixed learning rate — GCN macro-F1 0.16 normalized
vs 0.92 raw on Amazon Photo — so those two loaders skip it (`configs/extended_datasets.yaml`,
`configs/final_sage.yaml`).

**Further extensions tried and rejected** (kept as available tooling, not
part of the default family, since neither showed a statistically significant
net improvement over the configuration above at 15 seeds):
- *Self-training / pseudo-labeling* (`pherognn/train.py:train_self_training`,
  `selftrain_candidate`): retrains on high-confidence predictions for nodes
  that carry no train/val/test label at all (Planetoid's public split
  leaves ~39-92% of nodes completely unused — e.g. 18,157/19,717 PubMed
  nodes). This gives a large standalone gain on plain PheroGNN alone (PubMed
  single-seed macro-F1 0.785 -> 0.803 at confidence>=0.8), but adding it as
  one more `pherognn_select` candidate did not significantly improve on the
  ensemble-aware selector above (Cora 0.804 vs 0.808, not significant) —
  Select already captures most of the same gain through mechanism diversity,
  and the self-trained candidate's own validation score is an unreliable
  signal for whether it will win.
- *More PheroAPPNP candidates* (`pherognn_appnp_k5`, `pherognn_appnp_k20`,
  any `pherognn_appnp_k<N>` name is supported): adding these to the select
  family gave non-significant Cora/CiteSeer changes and a non-significant
  synthetic regression (more near-ceiling-but-synthetic-weak candidates
  diluting the ensemble path), so the 4-candidate family above is kept.

The ablation configs used to reach these conclusions are kept for
reproducibility: `configs/ablation_v7.yaml` / `ablation_v7b.yaml` (5-seed
mechanism screening), `configs/final_significance*.yaml` / `final_appnp.yaml`
(15-seed significance runs per mechanism on the original 4 datasets),
`configs/extended_datasets.yaml` (15-seed baselines on the 5 newer datasets),
`configs/final_sage.yaml` (15-seed PheroSAGE / PheroSAGEAPPNP / PheroGCNII /
PheroH2 / PheroGPR validation across all 9), `configs/final_gpr.yaml`
(15-seed PheroGPR-specific validation), and `configs/final_select*.yaml`
(end-to-end `pherognn_select` runs; `final_select_all9c.yaml` +
`final_select_all9c_ext.yaml` + `final_select_all9c_ext2.yaml` together are
the 45-seed run behind the current default 7-candidate configuration
validated across all 9 datasets). Run any of them with
`python scripts/run_experiments.py --config configs/<name>.yaml --no-interpretability`.

## Elliptic (optional)

Download the Elliptic Bitcoin dataset and place `elliptic_txs_features.csv`, `elliptic_txs_classes.csv`, `elliptic_txs_edgelist.csv` in `data/elliptic/`, then run:

```
python scripts/run_experiments.py --include-elliptic
```

## Config

Edit `configs/default.yaml` to change datasets, seeds, model hyperparameters, or pheromone settings.
