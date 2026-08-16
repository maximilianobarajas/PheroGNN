from __future__ import annotations
from typing import Dict
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


class PheroConv(MessagePassing):
    def __init__(self, in_dim, out_dim, tau_eps=1e-8):
        super().__init__(aggr="add", node_dim=0)
        self.lin = nn.Linear(in_dim, out_dim, bias=False)
        self.bias = nn.Parameter(torch.zeros(out_dim))
        self.tau_eps = float(tau_eps)

    def forward(self, x, edge_index, tau):
        num_nodes = x.size(0)
        num_edges = edge_index.size(1)

        edge_index_loop, _ = add_remaining_self_loops(edge_index, num_nodes=num_nodes)
        added = edge_index_loop.size(1) - num_edges
        if added:
            loop_tau = torch.ones(added, device=tau.device, dtype=tau.dtype)
            tau_loop = torch.cat([tau, loop_tau], dim=0)
        else:
            tau_loop = tau

        src, dst = edge_index_loop

        incoming_sum = scatter(tau_loop, dst, dim=0, dim_size=num_nodes, reduce="sum")
        incoming_count = scatter(
            torch.ones_like(tau_loop), dst, dim=0, dim_size=num_nodes, reduce="sum"
        ).clamp_min(1.0)
        incoming_mean = (incoming_sum / incoming_count).clamp_min(self.tau_eps)
        relative_tau = tau_loop / incoming_mean[dst]

        degree = scatter(
            torch.ones_like(relative_tau), dst, dim=0, dim_size=num_nodes, reduce="sum"
        ).clamp_min(1.0)
        degree_inv_sqrt = degree.pow(-0.5)
        gcn_norm = degree_inv_sqrt[src] * degree_inv_sqrt[dst]
        edge_weight = gcn_norm * relative_tau

        out = self.propagate(edge_index_loop, x=self.lin(x), edge_weight=edge_weight)
        return out + self.bias

    def message(self, x_j, edge_weight):
        return x_j * edge_weight.view(-1, 1)


class PheroGNN(nn.Module):
    def __init__(self, in_dim, hidden, out_dim, dropout, num_edges,
                 tau0=1.0, tau_min=0.05, tau_max=5.0,
                 evaporation_rate=0.08, reinforcement_rate=0.35,
                 credit_hops=1, credit_decay=0.5):
        super().__init__()
        self.c1 = PheroConv(in_dim, hidden)
        self.c2 = PheroConv(hidden, out_dim)
        self.dropout = float(dropout)

        self.tau0 = float(tau0)
        self.tau_min = float(tau_min)
        self.tau_max = float(tau_max)
        self.evaporation_rate = float(evaporation_rate)
        self.reinforcement_rate = float(reinforcement_rate)
        self.credit_hops = int(credit_hops)
        self.credit_decay = float(credit_decay)

        self.register_buffer("tau", torch.full((num_edges,), self.tau0))

    def forward(self, x, edge_index):
        x = self.c1(x, edge_index, self.tau).relu()
        x = F.dropout(x, self.dropout, self.training)
        return self.c2(x, edge_index, self.tau)

    @torch.no_grad()
    def update_pheromones(self, logits, y, train_mask, edge_index):
        if self.evaporation_rate > 0:
            self.tau.mul_(1.0 - self.evaporation_rate).add_(self.evaporation_rate * self.tau0)

        if self.reinforcement_rate > 0:
            probs = torch.softmax(logits, dim=-1)
            safe_y = y.clamp_min(0)
            true_prob = probs.gather(1, safe_y.unsqueeze(1)).squeeze(1)
            reward_node = torch.zeros(logits.size(0), device=logits.device, dtype=self.tau.dtype)
            reward_node[train_mask] = 2.0 * true_prob[train_mask] - 1.0

            src, dst = edge_index[0], edge_index[1]
            num_nodes = logits.size(0)
            current = reward_node
            total_edge_reward = torch.zeros_like(self.tau)
            for hop in range(max(1, self.credit_hops)):
                total_edge_reward = total_edge_reward + (self.credit_decay ** hop) * current[dst]
                if hop + 1 < self.credit_hops:
                    current = scatter(current[dst], src, dim=0, dim_size=num_nodes, reduce="mean")
            self.tau.add_(self.reinforcement_rate * total_edge_reward)

        self.tau.clamp_(self.tau_min, self.tau_max)

    def pheromone_statistics(self) -> Dict[str, float]:
        tau = self.tau.detach()
        return {
            "tau_mean": tau.mean().item(),
            "tau_std": tau.std(unbiased=False).item(),
            "tau_min": tau.min().item(),
            "tau_max": tau.max().item(),
        }


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
        evaporation_rate=pheromone_cfg.get("evaporation_rate", 0.08),
        reinforcement_rate=pheromone_cfg.get("reinforcement_rate", 0.35),
        credit_hops=pheromone_cfg.get("credit_hops", 1),
        credit_decay=pheromone_cfg.get("credit_decay", 0.5),
    )
