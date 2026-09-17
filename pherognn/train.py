from __future__ import annotations
from copy import deepcopy
import time
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from .models import PheroGNN, PheroGNNv7, PheroAPPNP

PHERO_MODELS = (PheroGNN, PheroGNNv7, PheroAPPNP)


def metrics(logits, y, mask, num_classes):
    yt = y[mask].detach().cpu().numpy()
    prob = torch.softmax(logits[mask], -1).detach().cpu().numpy()
    yp = prob.argmax(1)

    precision, recall, f1, _ = precision_recall_fscore_support(yt, yp, average="macro", zero_division=0)
    result = dict(
        accuracy=accuracy_score(yt, yp),
        macro_precision=precision,
        macro_recall=recall,
        macro_f1=f1,
        balanced_accuracy=balanced_accuracy_score(yt, yp),
    )

    if num_classes == 2 and len(np.unique(yt)) == 2:
        result["roc_auc"] = roc_auc_score(yt, prob[:, 1])
        result["pr_auc"] = average_precision_score(yt, prob[:, 1])
        p_bin, r_bin, f1_bin, _ = precision_recall_fscore_support(yt, yp, average="binary", pos_label=1, zero_division=0)
        result.update(fraud_precision=p_bin, fraud_recall=r_bin, fraud_f1=f1_bin)
    else:
        result.update(roc_auc=np.nan, pr_auc=np.nan, fraud_precision=np.nan, fraud_recall=np.nan, fraud_f1=np.nan)
    return result


def train_one(model, data, cfg):
    model_cfg = cfg["model"]
    experiment_cfg = cfg["experiment"]
    gradient_clip = cfg.get("pheromone", {}).get("gradient_clip", 5.0)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=float(model_cfg["learning_rate"]), weight_decay=float(model_cfg["weight_decay"])
    )

    best_score = -np.inf
    best_state = None
    best_epoch = 0
    stale = 0
    history = []
    start_time = time.perf_counter()

    for epoch in range(1, experiment_cfg["epochs"] + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)

        logits = model(data.x, data.edge_index)
        loss = F.cross_entropy(logits[data.train_mask], data.y[data.train_mask])
        loss.backward()

        if gradient_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(gradient_clip))

        optimizer.step()

        if isinstance(model, PHERO_MODELS):
            model.eval()
            with torch.no_grad():
                reward_logits = model(data.x, data.edge_index)
            model.update_pheromones(reward_logits, data.y, data.train_mask, data.edge_index)

        model.eval()
        with torch.no_grad():
            validation_logits = model(data.x, data.edge_index)

        validation = metrics(validation_logits, data.y, data.val_mask, data.num_classes)
        test = metrics(validation_logits, data.y, data.test_mask, data.num_classes)

        epoch_record = {
            "epoch": epoch,
            "loss": loss.detach().item(),
            **{f"val_{k}": v for k, v in validation.items()},
            **{f"test_{k}": v for k, v in test.items()},
        }
        if isinstance(model, PHERO_MODELS):
            epoch_record.update(model.pheromone_statistics())
        history.append(epoch_record)

        score = validation[experiment_cfg["selection_metric"]]
        if np.isfinite(score) and score > best_score:
            best_score = score
            best_state = deepcopy(model.state_dict())
            best_epoch = epoch
            stale = 0
        else:
            stale += 1

        if stale >= experiment_cfg["patience"]:
            break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        final_logits = model(data.x, data.edge_index)

    final = metrics(final_logits, data.y, data.test_mask, data.num_classes)
    final["best_epoch"] = best_epoch
    final["best_val_score"] = best_score
    final["runtime_seconds"] = time.perf_counter() - start_time
    if isinstance(model, PHERO_MODELS):
        final.update(model.pheromone_statistics())

    return model, pd.DataFrame(history), final


# Small family of pheromone-routing mechanisms (persistent pheromone alone,
# ACO heuristic-guided pheromone combined with a heterophily-pruning signal,
# structure-aware DropEdge, and decoupled APPNP-style deep pheromone
# diffusion) whose relative strength is dataset-dependent: see ablation
# results. PheroGNN-Select trains each candidate and picks the winner using
# only the validation partition, matching the paper's own protocol of
# validation-only dataset-specific tuning, so it is never worse than plain
# PheroGNN in expectation and captures genuine gains (e.g. Cora, CiteSeer,
# PubMed) when a mechanism's inductive bias matches the graph. pherognn_appnp
# significantly hurts the small synthetic fraud graph (too many diffusion
# hops oversmooths a 500-node graph) but significantly helps CiteSeer/PubMed,
# which is exactly the kind of dataset-dependent trade-off this selector is
# designed to resolve using only validation, never test, labels.
PHEROGNN_SELECT_FAMILY = [
    "pherognn", "pherognn_v7_heuristic_hetero", "pherognn_v7_dropedge", "pherognn_appnp",
]


def train_select(build_model_fn, data, cfg, family=None):
    family = family or PHEROGNN_SELECT_FAMILY
    candidates = {}
    for name in family:
        model = build_model_fn(name, data, cfg)
        model = model.to(data.x.device)
        model, history, final = train_one(model, data, cfg)
        candidates[name] = (model, history, final)

    winner_name = max(candidates, key=lambda n: candidates[n][2]["best_val_score"])
    model, history, final = candidates[winner_name]
    final = dict(final)
    final["selected_variant"] = winner_name
    return model, history, final
