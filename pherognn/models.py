from __future__ import annotations
from typing import Dict
import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import APPNP, GATConv, GCNConv, MessagePassing, SAGEConv
from torch_geometric.utils import add_remaining_self_loops, scatter
from torch_geometric.utils import softmax as scatter_softmax


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

    def effective_tau(self):
        return self.tau

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


class PheroConvV2(MessagePassing):
    """Extends PheroConv with an optional learned ACO-style heuristic term and
    optional structure-aware DropEdge, combined with the persistent pheromone trail."""

    def __init__(self, in_dim, out_dim, use_heuristic=False, alpha=1.0, beta=1.0,
                 learn_beta=False, tau_eps=1e-8):
        super().__init__(aggr="add", node_dim=0)
        self.lin = nn.Linear(in_dim, out_dim, bias=False)
        self.bias = nn.Parameter(torch.zeros(out_dim))
        self.tau_eps = float(tau_eps)
        self.use_heuristic = bool(use_heuristic)
        self.alpha = float(alpha)
        self.learn_beta = bool(learn_beta)
        if self.learn_beta:
            # Softplus-parameterized so the model can shrink the learned heuristic's
            # influence toward 0 (falling back to pure pheromone routing) when the
            # extra attention parameters would only overfit a label-scarce dataset.
            self.beta_raw = nn.Parameter(torch.tensor(float(beta)))
        else:
            self.beta = float(beta)
        if self.use_heuristic:
            self.att_src = nn.Linear(in_dim, 1, bias=False)
            self.att_dst = nn.Linear(in_dim, 1, bias=False)

    def effective_beta(self):
        return F.softplus(self.beta_raw) if self.learn_beta else self.beta

    def forward(self, x, edge_index, tau, dropedge_p=0.0):
        num_nodes = x.size(0)
        num_edges = edge_index.size(1)

        edge_index_loop, _ = add_remaining_self_loops(edge_index, num_nodes=num_nodes)
        added = edge_index_loop.size(1) - num_edges
        if added:
            loop_tau = torch.ones(added, device=tau.device, dtype=tau.dtype)
            tau_loop = torch.cat([tau, loop_tau], dim=0)
        else:
            tau_loop = tau

        if self.training and dropedge_p > 0:
            keep = torch.rand(edge_index_loop.size(1), device=x.device) >= dropedge_p
            keep[num_edges:] = True  # never drop self-loops
            edge_index_loop = edge_index_loop[:, keep]
            tau_loop = tau_loop[keep]

        src, dst = edge_index_loop
        h = self.lin(x)

        if self.use_heuristic:
            # Classic ACO transition rule p_ij ~ tau_ij^alpha * eta_ij^beta, with a
            # learned GAT-style heuristic eta guiding exploration jointly with the
            # persistent pheromone trail (inspired by graph-transformer-guided ACO).
            att_logit = self.att_src(x)[src].squeeze(-1) + self.att_dst(x)[dst].squeeze(-1)
            att_logit = F.leaky_relu(att_logit, 0.2)
            log_tau = torch.log(tau_loop.clamp_min(self.tau_eps))
            combined_logit = self.alpha * log_tau + self.effective_beta() * att_logit
            edge_weight = scatter_softmax(combined_logit, dst, num_nodes=num_nodes)
        else:
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

        out = self.propagate(edge_index_loop, x=h, edge_weight=edge_weight)
        return out + self.bias

    def message(self, x_j, edge_weight):
        return x_j * edge_weight.view(-1, 1)


class PheroGNNv7(nn.Module):
    """PheroGNN extended with mechanisms inspired by recent GNN/ACO literature:
    - use_heuristic: learned per-edge heuristic combined with pheromone via the
      classical ACO transition rule tau^alpha * eta^beta (graph-transformer-guided
      ACO / DeepACO style neural guidance).
    - dual_pheromone: separate positive (attractive) and negative (repellent)
      pheromone trails, the effective edge weight being their ratio (negative
      pheromones / dual-colony ACO).
    - heterophily_weight: unsupervised structural signal that reinforces the
      negative trail on edges whose endpoints currently disagree in predicted
      class, independent of the train mask (graph-rewiring / heterophily-pruning
      literature), pushing the model to suppress likely-heterophilic edges.
    - use_gate: a per-node learned gate mixing the 1-hop and 2-hop
      representations (node-wise prioritized/personalized propagation).
    - use_dropedge: layer-dependent structure-aware DropEdge regularization.
    """

    def __init__(self, in_dim, hidden, out_dim, dropout, num_edges,
                 tau0=1.0, tau_min=0.05, tau_max=5.0,
                 evaporation_rate=0.08, reinforcement_rate=0.35,
                 credit_hops=1, credit_decay=0.5,
                 use_heuristic=False, alpha=1.0, beta=1.0, learn_beta=False,
                 dual_pheromone=False, heterophily_weight=0.0,
                 use_gate=False, gate_hidden=16,
                 use_dropedge=False, dropedge_base_p=0.15, dropedge_layer_growth=0.5,
                 tau_eps=1e-8):
        super().__init__()
        self.c1 = PheroConvV2(in_dim, hidden, use_heuristic, alpha, beta, learn_beta, tau_eps)
        self.c2 = PheroConvV2(hidden, out_dim, use_heuristic, alpha, beta, learn_beta, tau_eps)
        self.dropout = float(dropout)

        self.tau0 = float(tau0)
        self.tau_min = float(tau_min)
        self.tau_max = float(tau_max)
        self.evaporation_rate = float(evaporation_rate)
        self.reinforcement_rate = float(reinforcement_rate)
        self.credit_hops = int(credit_hops)
        self.credit_decay = float(credit_decay)
        self.tau_eps = float(tau_eps)

        self.dual_pheromone = bool(dual_pheromone)
        self.heterophily_weight = float(heterophily_weight)
        self.use_gate = bool(use_gate)
        self.use_dropedge = bool(use_dropedge)
        self.dropedge_base_p = float(dropedge_base_p)
        self.dropedge_layer_growth = float(dropedge_layer_growth)

        if self.dual_pheromone:
            self.register_buffer("tau_pos", torch.full((num_edges,), self.tau0))
            self.register_buffer("tau_neg", torch.full((num_edges,), self.tau_min))
        else:
            self.register_buffer("tau", torch.full((num_edges,), self.tau0))

        if self.use_gate:
            self.skip_proj = nn.Linear(hidden, out_dim)
            self.gate_mlp = nn.Sequential(
                nn.Linear(hidden + out_dim, gate_hidden), nn.ReLU(), nn.Linear(gate_hidden, 1)
            )

    def effective_tau(self):
        if self.dual_pheromone:
            return (self.tau_pos / (self.tau_neg + self.tau_eps)).clamp(self.tau_min, self.tau_max)
        return self.tau

    def forward(self, x, edge_index):
        tau = self.effective_tau()
        p1 = self.dropedge_base_p if self.use_dropedge else 0.0
        p2 = self.dropedge_base_p * (1.0 + self.dropedge_layer_growth) if self.use_dropedge else 0.0

        h1 = self.c1(x, edge_index, tau, dropedge_p=p1).relu()
        h1_drop = F.dropout(h1, self.dropout, self.training)
        h2 = self.c2(h1_drop, edge_index, tau, dropedge_p=p2)

        if self.use_gate:
            gate = torch.sigmoid(self.gate_mlp(torch.cat([h1, h2], dim=-1)))
            out = gate * h2 + (1.0 - gate) * self.skip_proj(h1)
            return out
        return h2

    def _credit_edge_reward(self, reward_node, edge_index, like):
        src, dst = edge_index[0], edge_index[1]
        num_nodes = reward_node.size(0)
        current = reward_node
        total_edge_reward = torch.zeros_like(like)
        for hop in range(max(1, self.credit_hops)):
            total_edge_reward = total_edge_reward + (self.credit_decay ** hop) * current[dst]
            if hop + 1 < self.credit_hops:
                current = scatter(current[dst], src, dim=0, dim_size=num_nodes, reduce="mean")
        return total_edge_reward

    @torch.no_grad()
    def update_pheromones(self, logits, y, train_mask, edge_index):
        probs = torch.softmax(logits, dim=-1)
        safe_y = y.clamp_min(0)
        true_prob = probs.gather(1, safe_y.unsqueeze(1)).squeeze(1)
        reward_node = torch.zeros(logits.size(0), device=logits.device, dtype=self.effective_tau().dtype)
        reward_node[train_mask] = 2.0 * true_prob[train_mask] - 1.0

        src, dst = edge_index[0], edge_index[1]
        pred = logits.argmax(dim=-1)
        disagree_edge = (pred[src] != pred[dst]).to(dtype=probs.dtype)

        if self.dual_pheromone:
            if self.evaporation_rate > 0:
                self.tau_pos.mul_(1.0 - self.evaporation_rate).add_(self.evaporation_rate * self.tau0)
                self.tau_neg.mul_(1.0 - self.evaporation_rate).add_(self.evaporation_rate * self.tau_min)

            if self.reinforcement_rate > 0:
                pos_reward_node = reward_node.clamp_min(0.0)
                neg_reward_node = (-reward_node).clamp_min(0.0)
                pos_edge_reward = self._credit_edge_reward(pos_reward_node, edge_index, self.tau_pos)
                neg_edge_reward = self._credit_edge_reward(neg_reward_node, edge_index, self.tau_neg)
                self.tau_pos.add_(self.reinforcement_rate * pos_edge_reward)
                self.tau_neg.add_(self.reinforcement_rate * neg_edge_reward)

            if self.heterophily_weight > 0:
                self.tau_neg.add_(self.heterophily_weight * disagree_edge)

            self.tau_pos.clamp_(self.tau_min, self.tau_max)
            self.tau_neg.clamp_(self.tau_min, self.tau_max)
        else:
            if self.evaporation_rate > 0:
                self.tau.mul_(1.0 - self.evaporation_rate).add_(self.evaporation_rate * self.tau0)
            if self.reinforcement_rate > 0:
                edge_reward = self._credit_edge_reward(reward_node, edge_index, self.tau)
                self.tau.add_(self.reinforcement_rate * edge_reward)
            self.tau.clamp_(self.tau_min, self.tau_max)

    def pheromone_statistics(self) -> Dict[str, float]:
        if self.dual_pheromone:
            tau_eff = self.effective_tau().detach()
            stats = {
                "tau_mean": tau_eff.mean().item(),
                "tau_std": tau_eff.std(unbiased=False).item(),
                "tau_min": tau_eff.min().item(),
                "tau_max": tau_eff.max().item(),
                "tau_pos_mean": self.tau_pos.mean().item(),
                "tau_neg_mean": self.tau_neg.mean().item(),
            }
            return stats
        tau = self.tau.detach()
        return {
            "tau_mean": tau.mean().item(),
            "tau_std": tau.std(unbiased=False).item(),
            "tau_min": tau.min().item(),
            "tau_max": tau.max().item(),
        }


class PheroConvSAGE(MessagePassing):
    """GraphSAGE-style layer (separate self/neighbor transforms, mean
    aggregation) with neighbor messages weighted by relative pheromone,
    instead of PheroConv's GCN-style symmetric-normalized sum aggregation.
    Motivated by an empirical finding: on strongly heterophilic graphs
    (Texas/Wisconsin/Cornell) and on larger co-purchase/co-authorship graphs
    (Amazon Photo, Coauthor CS), plain GraphSAGE significantly outperforms
    every PheroGNN variant built on PheroConv's GCN-style backbone -- the
    persistent pheromone memory does not, by itself, fix the structural
    disadvantage of symmetric-normalized aggregation on those graphs."""

    def __init__(self, in_dim, out_dim, use_heuristic=False, beta=0.5, tau_eps=1e-8):
        super().__init__(aggr="mean", node_dim=0)
        self.lin_self = nn.Linear(in_dim, out_dim, bias=True)
        self.lin_neigh = nn.Linear(in_dim, out_dim, bias=False)
        self.tau_eps = float(tau_eps)
        self.use_heuristic = bool(use_heuristic)
        self.beta = float(beta)
        if self.use_heuristic:
            self.att_src = nn.Linear(in_dim, 1, bias=False)
            self.att_dst = nn.Linear(in_dim, 1, bias=False)

    def forward(self, x, edge_index, tau):
        num_nodes = x.size(0)
        src, dst = edge_index

        incoming_sum = scatter(tau, dst, dim=0, dim_size=num_nodes, reduce="sum")
        incoming_count = scatter(
            torch.ones_like(tau), dst, dim=0, dim_size=num_nodes, reduce="sum"
        ).clamp_min(1.0)
        incoming_mean = (incoming_sum / incoming_count).clamp_min(self.tau_eps)
        relative_tau = tau / incoming_mean[dst]

        if self.use_heuristic:
            # ACO-style tau^1 * eta^beta combination (as in PheroConvV2's
            # use_heuristic path) but kept on SAGE's mean-aggregation scale
            # (normalized to mean ~1 per neighborhood) instead of a softmax,
            # so it still composes with separate self/neighbor transforms.
            att_logit = self.att_src(x)[src].squeeze(-1) + self.att_dst(x)[dst].squeeze(-1)
            eta = F.softplus(F.leaky_relu(att_logit, 0.2)) + self.tau_eps
            incoming_eta_sum = scatter(eta, dst, dim=0, dim_size=num_nodes, reduce="sum")
            incoming_eta_mean = (incoming_eta_sum / incoming_count).clamp_min(self.tau_eps)
            relative_eta = eta / incoming_eta_mean[dst]
            edge_weight = relative_tau * relative_eta.pow(self.beta)
        else:
            edge_weight = relative_tau

        neigh = self.propagate(edge_index, x=self.lin_neigh(x), edge_weight=edge_weight)
        return self.lin_self(x) + neigh

    def message(self, x_j, edge_weight):
        return x_j * edge_weight.view(-1, 1)


class PheroSAGE(nn.Module):
    def __init__(self, in_dim, hidden, out_dim, dropout, num_edges,
                 tau0=1.0, tau_min=0.05, tau_max=5.0,
                 evaporation_rate=0.08, reinforcement_rate=0.35,
                 credit_hops=1, credit_decay=0.5,
                 use_heuristic=False, beta=0.5):
        super().__init__()
        self.c1 = PheroConvSAGE(in_dim, hidden, use_heuristic, beta)
        self.c2 = PheroConvSAGE(hidden, out_dim, use_heuristic, beta)
        self.dropout = float(dropout)

        self.tau0 = float(tau0)
        self.tau_min = float(tau_min)
        self.tau_max = float(tau_max)
        self.evaporation_rate = float(evaporation_rate)
        self.reinforcement_rate = float(reinforcement_rate)
        self.credit_hops = int(credit_hops)
        self.credit_decay = float(credit_decay)

        self.register_buffer("tau", torch.full((num_edges,), self.tau0))

    def effective_tau(self):
        return self.tau

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


class PheroAPPNP(nn.Module):
    """Decouples feature transformation from propagation (Predict-then-Propagate
    / APPNP): a shallow 2-layer MLP produces an initial prediction, which is
    then diffused K hops through the persistent-pheromone-weighted graph with
    a teleport back to the initial prediction at every hop. This lets the
    model use a much larger receptive field than plain 2-layer PheroGNN
    without adding learnable graph layers (and therefore without the
    oversmoothing/overfitting risk of stacking more message-passing layers),
    while the pheromone trail governing that diffusion is still updated by
    the same evaporate/reinforce ant-colony dynamics as PheroGNN v6.
    """

    def __init__(self, in_dim, hidden, out_dim, dropout, num_edges,
                 tau0=1.0, tau_min=0.05, tau_max=5.0,
                 evaporation_rate=0.08, reinforcement_rate=0.35,
                 credit_hops=1, credit_decay=0.5,
                 K=10, appnp_alpha=0.1, tau_eps=1e-8):
        super().__init__()
        self.lin1 = nn.Linear(in_dim, hidden)
        self.lin2 = nn.Linear(hidden, out_dim)
        self.dropout = float(dropout)
        self.prop = APPNP(K=int(K), alpha=float(appnp_alpha), cached=False,
                           add_self_loops=True, normalize=True)

        self.tau0 = float(tau0)
        self.tau_min = float(tau_min)
        self.tau_max = float(tau_max)
        self.evaporation_rate = float(evaporation_rate)
        self.reinforcement_rate = float(reinforcement_rate)
        self.credit_hops = int(credit_hops)
        self.credit_decay = float(credit_decay)
        self.tau_eps = float(tau_eps)

        self.register_buffer("tau", torch.full((num_edges,), self.tau0))

    def effective_tau(self):
        return self.tau

    def forward(self, x, edge_index):
        h = F.relu(self.lin1(x))
        h = F.dropout(h, self.dropout, self.training)
        h = self.lin2(h)
        h = F.dropout(h, self.dropout, self.training)
        return self.prop(h, edge_index, edge_weight=self.tau)

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


class PheroEnsemble(nn.Module):
    """Uniform probability-space ensemble over a small set of already-trained
    PheroGNN-family candidates. Ablation showed hard "pick one winner by
    validation" beats averaging on CiteSeer (because one candidate,
    PheroAPPNP, is uniquely strong there and dilution hurts) but averaging
    beats picking a single winner on Cora and PubMed (decorrelated errors
    across mechanisms). `train_select` resolves this by treating this
    ensemble as one more validation-scored candidate alongside each
    individual member, so the choice between "one winner" and "average of
    all" is itself made on the validation partition only."""

    def __init__(self, members):
        super().__init__()
        self.members = nn.ModuleList(members)

    def forward(self, x, edge_index):
        probs = torch.stack(
            [F.softmax(m(x, edge_index), dim=-1) for m in self.members], dim=0
        ).mean(dim=0)
        return torch.log(probs.clamp_min(1e-12))


class PheroSAGEAPPNP(nn.Module):
    """Hybrid of PheroAPPNP and PheroSAGE: a shallow 2-layer MLP produces an
    initial prediction, which is then diffused K hops with a teleport back to
    it at every hop (as in PheroAPPNP) -- but through a row-stochastic
    (mean-aggregation, GraphSAGE-style) pheromone-weighted propagation
    instead of PheroAPPNP's GCN-style symmetric-normalized one. The idea is
    to combine APPNP's larger receptive field (helps CiteSeer/PubMed) with
    mean-aggregation's robustness to heterophily (helps Texas/Wisconsin/
    Cornell, where PheroAPPNP's GCN-style propagation underperforms
    GraphSAGE).
    """

    def __init__(self, in_dim, hidden, out_dim, dropout, num_edges,
                 tau0=1.0, tau_min=0.05, tau_max=5.0,
                 evaporation_rate=0.08, reinforcement_rate=0.35,
                 credit_hops=1, credit_decay=0.5,
                 K=10, appnp_alpha=0.1, tau_eps=1e-8):
        super().__init__()
        self.lin1 = nn.Linear(in_dim, hidden)
        self.lin2 = nn.Linear(hidden, out_dim)
        self.dropout = float(dropout)
        self.K = int(K)
        self.appnp_alpha = float(appnp_alpha)
        self.tau_eps = float(tau_eps)

        self.tau0 = float(tau0)
        self.tau_min = float(tau_min)
        self.tau_max = float(tau_max)
        self.evaporation_rate = float(evaporation_rate)
        self.reinforcement_rate = float(reinforcement_rate)
        self.credit_hops = int(credit_hops)
        self.credit_decay = float(credit_decay)

        self.register_buffer("tau", torch.full((num_edges,), self.tau0))

    def effective_tau(self):
        return self.tau

    def _mean_propagate(self, z, edge_index, tau):
        num_nodes = z.size(0)
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

        msg = z[src] * relative_tau.view(-1, 1)
        return scatter(msg, dst, dim=0, dim_size=num_nodes, reduce="mean")

    def forward(self, x, edge_index):
        h = F.relu(self.lin1(x))
        h = F.dropout(h, self.dropout, self.training)
        h0 = self.lin2(h)
        h0 = F.dropout(h0, self.dropout, self.training)

        z = h0
        for _ in range(self.K):
            z = self._mean_propagate(z, edge_index, self.tau)
            z = (1.0 - self.appnp_alpha) * z + self.appnp_alpha * h0
        return z

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


class PheroGCNIIConv(MessagePassing):
    """One GCNII layer (Chen et al., ICML 2020): initial-residual connection
    to the first-layer representation plus an identity-mapping-biased linear
    transform, which together let GNNs go much deeper (8+ layers) without
    oversmoothing -- unlike plain stacked GCNConv/PheroConv layers. The
    propagation step uses the same pheromone-weighted GCN-style
    normalization as PheroConv."""

    def __init__(self, dim, alpha, beta, tau_eps=1e-8):
        super().__init__(aggr="add", node_dim=0)
        self.weight = nn.Linear(dim, dim, bias=False)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.tau_eps = float(tau_eps)

    def forward(self, x, x0, edge_index, tau):
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

        propagated = self.propagate(edge_index_loop, x=x, edge_weight=edge_weight)
        h = (1.0 - self.alpha) * propagated + self.alpha * x0
        return (1.0 - self.beta) * h + self.beta * self.weight(h)

    def message(self, x_j, edge_weight):
        return x_j * edge_weight.view(-1, 1)


class PheroGCNII(nn.Module):
    """Deep PheroGNN using GCNII's initial-residual + identity-mapping trick
    to go to `num_layers` (default 16) layers without oversmoothing, with
    every layer's propagation weighted by the same persistent, evaporating/
    reinforcing pheromone trail as the rest of the PheroGNN family. Tests
    whether depth -- orthogonal to every mechanism tried so far, all of
    which kept PheroConv's original 2-layer depth -- is itself a lever the
    pheromone mechanism can exploit."""

    def __init__(self, in_dim, hidden, out_dim, dropout, num_edges,
                 tau0=1.0, tau_min=0.05, tau_max=5.0,
                 evaporation_rate=0.08, reinforcement_rate=0.35,
                 credit_hops=1, credit_decay=0.5,
                 num_layers=16, gcnii_alpha=0.1, gcnii_lambda=0.5, tau_eps=1e-8):
        super().__init__()
        self.lin_in = nn.Linear(in_dim, hidden)
        self.lin_out = nn.Linear(hidden, out_dim)
        self.dropout = float(dropout)
        self.convs = nn.ModuleList([
            PheroGCNIIConv(hidden, gcnii_alpha, min(1.0, gcnii_lambda / (layer + 1)), tau_eps)
            for layer in range(int(num_layers))
        ])

        self.tau0 = float(tau0)
        self.tau_min = float(tau_min)
        self.tau_max = float(tau_max)
        self.evaporation_rate = float(evaporation_rate)
        self.reinforcement_rate = float(reinforcement_rate)
        self.credit_hops = int(credit_hops)
        self.credit_decay = float(credit_decay)

        self.register_buffer("tau", torch.full((num_edges,), self.tau0))

    def effective_tau(self):
        return self.tau

    def forward(self, x, edge_index):
        x = F.dropout(x, self.dropout, self.training)
        x0 = F.relu(self.lin_in(x))
        x0 = F.dropout(x0, self.dropout, self.training)
        h = x0
        for conv in self.convs:
            h = conv(h, x0, edge_index, self.tau).relu()
            h = F.dropout(h, self.dropout, self.training)
        return self.lin_out(h)

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


PHEROGNN_V7_VARIANTS = {
    "pherognn_v7_heuristic": dict(use_heuristic=True),
    "pherognn_v7_dual": dict(dual_pheromone=True),
    "pherognn_v7_hetero": dict(dual_pheromone=True, heterophily_weight=0.02),
    "pherognn_v7_gate": dict(use_gate=True),
    "pherognn_v7_dropedge": dict(use_dropedge=True),
    "pherognn_v7_full": dict(
        use_heuristic=True, dual_pheromone=True, heterophily_weight=0.02, use_gate=True
    ),
    "pherognn_v7_heuristic_dropedge": dict(use_heuristic=True, use_dropedge=True),
    "pherognn_v7_heuristic_hetero": dict(
        use_heuristic=True, dual_pheromone=True, heterophily_weight=0.02
    ),
    "pherognn_v7_heuristic_dropedge_hetero": dict(
        use_heuristic=True, use_dropedge=True, dual_pheromone=True, heterophily_weight=0.02
    ),
    "pherognn_v7_heuristic_adaptive": dict(use_heuristic=True, learn_beta=True, beta=0.1),
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

    if name == "pherognn":
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

    if name in {"pherognn_sage", "pherognn_sage_heuristic"}:
        return PheroSAGE(
            *common,
            num_edges=data.edge_index.size(1),
            tau0=pheromone_cfg.get("tau0", 1.0),
            tau_min=pheromone_cfg.get("tau_min", 0.05),
            tau_max=pheromone_cfg.get("tau_max", 5.0),
            evaporation_rate=pheromone_cfg.get("evaporation_rate", 0.08),
            reinforcement_rate=pheromone_cfg.get("reinforcement_rate", 0.35),
            credit_hops=pheromone_cfg.get("credit_hops", 1),
            credit_decay=pheromone_cfg.get("credit_decay", 0.5),
            use_heuristic=(name == "pherognn_sage_heuristic"),
            beta=cfg.get("pheromone_v7", {}).get("beta", 0.5),
        )

    if name == "pherognn_sage_appnp":
        appnp_cfg = cfg.get("pheromone_appnp", {})
        return PheroSAGEAPPNP(
            *common,
            num_edges=data.edge_index.size(1),
            tau0=appnp_cfg.get("tau0", pheromone_cfg.get("tau0", 1.0)),
            tau_min=appnp_cfg.get("tau_min", pheromone_cfg.get("tau_min", 0.05)),
            tau_max=appnp_cfg.get("tau_max", pheromone_cfg.get("tau_max", 5.0)),
            evaporation_rate=appnp_cfg.get("evaporation_rate", pheromone_cfg.get("evaporation_rate", 0.08)),
            reinforcement_rate=appnp_cfg.get("reinforcement_rate", pheromone_cfg.get("reinforcement_rate", 0.35)),
            credit_hops=appnp_cfg.get("credit_hops", pheromone_cfg.get("credit_hops", 1)),
            credit_decay=appnp_cfg.get("credit_decay", pheromone_cfg.get("credit_decay", 0.5)),
            K=appnp_cfg.get("K", 10),
            appnp_alpha=appnp_cfg.get("appnp_alpha", 0.1),
        )

    if name == "pherognn_gcnii":
        gcnii_cfg = cfg.get("pheromone_gcnii", {})
        return PheroGCNII(
            *common,
            num_edges=data.edge_index.size(1),
            tau0=gcnii_cfg.get("tau0", pheromone_cfg.get("tau0", 1.0)),
            tau_min=gcnii_cfg.get("tau_min", pheromone_cfg.get("tau_min", 0.05)),
            tau_max=gcnii_cfg.get("tau_max", pheromone_cfg.get("tau_max", 5.0)),
            evaporation_rate=gcnii_cfg.get("evaporation_rate", pheromone_cfg.get("evaporation_rate", 0.08)),
            reinforcement_rate=gcnii_cfg.get("reinforcement_rate", pheromone_cfg.get("reinforcement_rate", 0.35)),
            credit_hops=gcnii_cfg.get("credit_hops", pheromone_cfg.get("credit_hops", 1)),
            credit_decay=gcnii_cfg.get("credit_decay", pheromone_cfg.get("credit_decay", 0.5)),
            num_layers=gcnii_cfg.get("num_layers", 16),
            gcnii_alpha=gcnii_cfg.get("gcnii_alpha", 0.1),
            gcnii_lambda=gcnii_cfg.get("gcnii_lambda", 0.5),
        )

    if name in PHEROGNN_V7_VARIANTS:
        v7_cfg = cfg.get("pheromone_v7", {})
        kwargs = dict(
            tau0=v7_cfg.get("tau0", pheromone_cfg.get("tau0", 1.0)),
            tau_min=v7_cfg.get("tau_min", pheromone_cfg.get("tau_min", 0.05)),
            tau_max=v7_cfg.get("tau_max", pheromone_cfg.get("tau_max", 5.0)),
            evaporation_rate=v7_cfg.get("evaporation_rate", pheromone_cfg.get("evaporation_rate", 0.08)),
            reinforcement_rate=v7_cfg.get("reinforcement_rate", pheromone_cfg.get("reinforcement_rate", 0.35)),
            credit_hops=v7_cfg.get("credit_hops", pheromone_cfg.get("credit_hops", 1)),
            credit_decay=v7_cfg.get("credit_decay", pheromone_cfg.get("credit_decay", 0.5)),
            alpha=v7_cfg.get("alpha", 1.0),
            beta=v7_cfg.get("beta", 1.0),
            gate_hidden=v7_cfg.get("gate_hidden", 16),
            dropedge_base_p=v7_cfg.get("dropedge_base_p", 0.15),
            dropedge_layer_growth=v7_cfg.get("dropedge_layer_growth", 0.5),
        )
        kwargs.update(PHEROGNN_V7_VARIANTS[name])
        return PheroGNNv7(*common, num_edges=data.edge_index.size(1), **kwargs)

    if name.startswith("pherognn_appnp"):
        appnp_cfg = cfg.get("pheromone_appnp", {})
        K = appnp_cfg.get("K", 10)
        appnp_alpha = appnp_cfg.get("appnp_alpha", 0.1)
        # allows dataset-dependent K to be exposed as separate named
        # candidates (e.g. "pherognn_appnp_k20") for PheroGNN-Select to
        # choose between via validation, without a config file per variant.
        if name != "pherognn_appnp":
            suffix = name[len("pherognn_appnp_"):]
            if suffix.startswith("k"):
                K = int(suffix[1:])
        return PheroAPPNP(
            *common,
            num_edges=data.edge_index.size(1),
            tau0=appnp_cfg.get("tau0", pheromone_cfg.get("tau0", 1.0)),
            tau_min=appnp_cfg.get("tau_min", pheromone_cfg.get("tau_min", 0.05)),
            tau_max=appnp_cfg.get("tau_max", pheromone_cfg.get("tau_max", 5.0)),
            evaporation_rate=appnp_cfg.get("evaporation_rate", pheromone_cfg.get("evaporation_rate", 0.08)),
            reinforcement_rate=appnp_cfg.get("reinforcement_rate", pheromone_cfg.get("reinforcement_rate", 0.35)),
            credit_hops=appnp_cfg.get("credit_hops", pheromone_cfg.get("credit_hops", 1)),
            credit_decay=appnp_cfg.get("credit_decay", pheromone_cfg.get("credit_decay", 0.5)),
            K=K,
            appnp_alpha=appnp_alpha,
        )

    raise ValueError(f"Unknown model: {name}")
