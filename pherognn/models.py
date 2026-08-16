from __future__ import annotations
import math
from typing import Dict, Optional
import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv, MessagePassing, SAGEConv
from torch_geometric.utils import add_remaining_self_loops, scatter


class GCN(nn.Module):
    def __init__(self, in_dim, hidden, out_dim, dropout):
        super().__init__()
        self.c1 = GCNConv(in_dim, hidden, cached=False)
        self.c2 = GCNConv(hidden, out_dim, cached=False)
        self.dropout = float(dropout)

    def forward(self, x, edge_index):
        x = self.c1(x, edge_index).relu()
        x = F.dropout(x, self.dropout, self.training)
        return self.c2(x, edge_index)


class GAT(nn.Module):
    def __init__(self, in_dim, hidden, out_dim, dropout, heads=8):
        super().__init__()
        per_head = max(1, hidden // heads)
        self.c1 = GATConv(in_dim, per_head, heads=heads, dropout=dropout)
        self.c2 = GATConv(per_head * heads, out_dim, heads=1, concat=False, dropout=dropout)
        self.dropout = float(dropout)

    def forward(self, x, edge_index):
        x = F.elu(self.c1(x, edge_index))
        x = F.dropout(x, self.dropout, self.training)
        return self.c2(x, edge_index)


class GraphSAGE(nn.Module):
    def __init__(self, in_dim, hidden, out_dim, dropout):
        super().__init__()
        self.c1 = SAGEConv(in_dim, hidden)
        self.c2 = SAGEConv(hidden, out_dim)
        self.dropout = float(dropout)

    def forward(self, x, edge_index):
        x = self.c1(x, edge_index).relu()
        x = F.dropout(x, self.dropout, self.training)
        return self.c2(x, edge_index)


def _inverse_softplus(value: float) -> float:
    value = max(float(value), 1e-8)
    if value > 20.0:
        return value
    return math.log(math.expm1(value))


class ResidualGatePheroConv(MessagePassing):
    def __init__(self, in_dim, out_dim, memory_dim, gate_hidden=16, gate_mlp_hidden=32,
                 max_alpha=1.0, min_edge_scale=0.05, max_edge_scale=2.0, tau_eps=1e-8):
        super().__init__(aggr="add", node_dim=0)
        self.lin = nn.Linear(in_dim, out_dim, bias=False)
        self.bias = nn.Parameter(torch.zeros(out_dim))
        self.memory_dim = int(memory_dim)
        self.max_alpha = float(max_alpha)
        self.min_edge_scale = float(min_edge_scale)
        self.max_edge_scale = float(max_edge_scale)
        self.tau_eps = float(tau_eps)

        self.source_gate_projection = nn.Linear(in_dim, gate_hidden, bias=False)
        self.target_gate_projection = nn.Linear(in_dim, gate_hidden, bias=False)
        gate_input_dim = self.memory_dim + 2 * gate_hidden

        self.gate_mlp = nn.Sequential(
            nn.Linear(gate_input_dim, gate_mlp_hidden),
            nn.ReLU(),
            nn.Linear(gate_mlp_hidden, 1),
        )
        self.alpha_raw = nn.Parameter(torch.zeros(()))
        nn.init.xavier_uniform_(self.gate_mlp[0].weight)
        nn.init.zeros_(self.gate_mlp[0].bias)
        nn.init.normal_(self.gate_mlp[2].weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.gate_mlp[2].bias)

        self._last_gate_score: Optional[torch.Tensor] = None
        self._last_edge_scale: Optional[torch.Tensor] = None

    @property
    def alpha(self) -> torch.Tensor:
        return self.max_alpha * torch.tanh(self.alpha_raw)

    @property
    def last_gate_score(self) -> Optional[torch.Tensor]:
        return self._last_gate_score

    @property
    def last_edge_scale(self) -> Optional[torch.Tensor]:
        return self._last_edge_scale

    def _relative_memory(self, tau_vector, dst, num_nodes):
        incoming_sum = scatter(tau_vector, dst, dim=0, dim_size=num_nodes, reduce="sum")
        incoming_count = scatter(
            torch.ones((tau_vector.size(0), 1), device=tau_vector.device, dtype=tau_vector.dtype),
            dst, dim=0, dim_size=num_nodes, reduce="sum",
        ).clamp_min(1.0)
        incoming_mean = (incoming_sum / incoming_count).clamp_min(self.tau_eps)
        return tau_vector / incoming_mean[dst] - 1.0

    def forward(self, x, edge_index, tau_vector):
        num_nodes = x.size(0)
        num_edges = edge_index.size(1)
        src, dst = edge_index
        relative_tau = self._relative_memory(tau_vector, dst, num_nodes)

        gate_inputs = [
            relative_tau,
            self.source_gate_projection(x)[src],
            self.target_gate_projection(x)[dst],
        ]
        gate_score = torch.tanh(self.gate_mlp(torch.cat(gate_inputs, dim=-1)).squeeze(-1))
        edge_scale = (1.0 + self.alpha * gate_score).clamp(self.min_edge_scale, self.max_edge_scale)

        self._last_gate_score = gate_score.detach()
        self._last_edge_scale = edge_scale.detach()

        edge_index_loop, _ = add_remaining_self_loops(edge_index, num_nodes=num_nodes)
        added_loops = edge_index_loop.size(1) - num_edges
        if added_loops:
            loop_scale = torch.ones(added_loops, device=edge_scale.device, dtype=edge_scale.dtype)
            scale_loop = torch.cat([edge_scale, loop_scale], dim=0)
        else:
            scale_loop = edge_scale

        src_loop, dst_loop = edge_index_loop
        degree = scatter(torch.ones_like(scale_loop), dst_loop, dim=0, dim_size=num_nodes, reduce="sum").clamp_min(1.0)
        degree_inv_sqrt = degree.pow(-0.5)
        gcn_norm = degree_inv_sqrt[src_loop] * degree_inv_sqrt[dst_loop]
        edge_weight = gcn_norm * scale_loop

        out = self.propagate(edge_index_loop, x=self.lin(x), edge_weight=edge_weight)
        return out + self.bias

    def message(self, x_j, edge_weight):
        return x_j * edge_weight.view(-1, 1)

    def gate_statistics(self) -> Dict[str, float]:
        if self._last_gate_score is None:
            return {}
        score = self._last_gate_score
        scale = self._last_edge_scale
        return {
            "alpha": float(self.alpha.detach().cpu()),
            "gate_mean": float(score.mean().cpu()),
            "gate_std": float(score.std(unbiased=False).cpu()),
            "edge_scale_mean": float(scale.mean().cpu()),
            "edge_scale_std": float(scale.std(unbiased=False).cpu()),
            "edge_scale_min": float(scale.min().cpu()),
            "edge_scale_max": float(scale.max().cpu()),
        }


class PheroGNN(nn.Module):
    def __init__(self, in_dim, hidden, out_dim, dropout, num_edges,
                 tau0=1.0, tau_min=0.05, tau_max=5.0, memory_dim=8,
                 gate_hidden=16, gate_mlp_hidden=32, max_alpha=1.0,
                 min_edge_scale=0.05, max_edge_scale=2.0):
        super().__init__()
        self.memory_dim = int(memory_dim)
        self.dropout = float(dropout)
        self.tau_min = float(tau_min)
        self.tau_max = float(tau_max)

        layer_kwargs = dict(
            memory_dim=self.memory_dim, gate_hidden=int(gate_hidden),
            gate_mlp_hidden=int(gate_mlp_hidden), max_alpha=float(max_alpha),
            min_edge_scale=float(min_edge_scale), max_edge_scale=float(max_edge_scale),
        )
        self.c1 = ResidualGatePheroConv(in_dim, hidden, **layer_kwargs)
        self.c2 = ResidualGatePheroConv(hidden, out_dim, **layer_kwargs)

        tau_raw = torch.full((num_edges, self.memory_dim), _inverse_softplus(tau0))
        self.tau_raw = nn.Parameter(tau_raw)

    @property
    def tau(self):
        return F.softplus(self.tau_raw).clamp(self.tau_min, self.tau_max)

    def forward(self, x, edge_index):
        tau = self.tau
        x = self.c1(x, edge_index, tau).relu()
        x = F.dropout(x, self.dropout, self.training)
        return self.c2(x, edge_index, tau)

    def pheromone_regularization(self):
        return (self.tau - 1.0).abs().mean()

    def gate_statistics(self) -> Dict[str, float]:
        statistics: Dict[str, float] = {}
        for layer_index, layer in enumerate((self.c1, self.c2), start=1):
            for key, value in layer.gate_statistics().items():
                statistics[f"layer{layer_index}_{key}"] = value
        return statistics

    @torch.no_grad()
    def clamp_pheromones_(self):
        self.tau_raw.clamp_(_inverse_softplus(self.tau_min), _inverse_softplus(self.tau_max))


def build_model(name, data, cfg):
    model_cfg = cfg["model"]
    pheromone_cfg = cfg.get("pheromone", {})
    common = (data.num_node_features, model_cfg["hidden_channels"], data.num_classes, model_cfg["dropout"])

    if name == "gcn":
        return GCN(*common)
    if name == "gat":
        return GAT(*common)
    if name == "graphsage":
        return GraphSAGE(*common)
    if name != "pherognn":
        raise ValueError(f"Unknown model: {name}")

    return PheroGNN(
        *common,
        num_edges=data.edge_index.size(1),
        tau0=pheromone_cfg.get("tau0", 1.0),
        tau_min=pheromone_cfg.get("tau_min", 0.05),
        tau_max=pheromone_cfg.get("tau_max", 5.0),
        memory_dim=pheromone_cfg.get("memory_dim", 8),
        gate_hidden=pheromone_cfg.get("gate_hidden", 16),
        gate_mlp_hidden=pheromone_cfg.get("gate_mlp_hidden", 32),
        max_alpha=pheromone_cfg.get("max_alpha", 1.0),
        min_edge_scale=pheromone_cfg.get("min_edge_scale", 0.05),
        max_edge_scale=pheromone_cfg.get("max_edge_scale", 2.0),
    )
