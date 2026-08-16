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
from .models import PheroGNN


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


def _build_optimizer(model, model_cfg, pheromone_cfg):
    base_lr = float(model_cfg["learning_rate"])
    weight_decay = float(model_cfg["weight_decay"])

    if not isinstance(model, PheroGNN):
        return torch.optim.Adam(model.parameters(), lr=base_lr, weight_decay=weight_decay)

    tau_lr = float(pheromone_cfg.get("learning_rate", base_lr * 0.5))
    tau_ids = {id(model.tau_raw)}
    network_parameters = [p for p in model.parameters() if id(p) not in tau_ids]

    return torch.optim.Adam([
        {"params": network_parameters, "lr": base_lr, "weight_decay": weight_decay},
        {"params": [model.tau_raw], "lr": tau_lr, "weight_decay": 0.0},
    ])


def _pheromone_statistics(model):
    tau = model.tau.detach()
    result = dict(
        tau_mean=tau.mean().item(),
        tau_std=tau.std(unbiased=False).item(),
        tau_min=tau.min().item(),
        tau_max=tau.max().item(),
    )
    result.update(model.gate_statistics())
    return result


def train_one(model, data, cfg):
    model_cfg = cfg["model"]
    experiment_cfg = cfg["experiment"]
    pheromone_cfg = cfg.get("pheromone", {})

    optimizer = _build_optimizer(model, model_cfg, pheromone_cfg)
    tau_l1 = float(pheromone_cfg.get("tau_l1", 1e-5))
    gradient_clip = pheromone_cfg.get("gradient_clip", 5.0)

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
        classification_loss = F.cross_entropy(logits[data.train_mask], data.y[data.train_mask])

        tau_loss = classification_loss.new_zeros(())
        if isinstance(model, PheroGNN) and tau_l1 > 0:
            tau_loss = tau_l1 * model.pheromone_regularization()

        loss = classification_loss + tau_loss
        loss.backward()

        if gradient_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(gradient_clip))

        optimizer.step()
        if isinstance(model, PheroGNN):
            model.clamp_pheromones_()

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
        if isinstance(model, PheroGNN):
            epoch_record.update(_pheromone_statistics(model))
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
    final["runtime_seconds"] = time.perf_counter() - start_time
    if isinstance(model, PheroGNN):
        final.update(_pheromone_statistics(model))

    return model, pd.DataFrame(history), final
