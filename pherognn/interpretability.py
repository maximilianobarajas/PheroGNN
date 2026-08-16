from __future__ import annotations
import numpy as np
import pandas as pd
import networkx as nx
import torch
from scipy.stats import pearsonr, spearmanr
from .models import PheroGNN


def edge_table(model, data):
    if not isinstance(model, PheroGNN):
        raise TypeError("Interpretability requires PheroGNN.")

    model.eval()
    with torch.no_grad():
        model(data.x, data.edge_index)

    src, dst = data.edge_index.detach().cpu().numpy()
    tau = model.tau.detach().cpu().numpy()
    if tau.ndim == 1:
        tau = tau[:, None]

    table = {
        "source": src,
        "target": dst,
        "pheromone": tau.mean(axis=1),
        "pheromone_l2": np.linalg.norm(tau, axis=1),
        "pheromone_std": tau.std(axis=1),
    }
    for channel in range(tau.shape[1]):
        table[f"pheromone_ch{channel}"] = tau[:, channel]

    for layer_index, layer in enumerate((model.c1, model.c2), start=1):
        gate = layer.last_gate_score
        scale = layer.last_edge_scale
        if gate is not None and len(gate) == len(src):
            table[f"gate_layer{layer_index}"] = gate.cpu().numpy()
            table[f"edge_scale_layer{layer_index}"] = scale.cpu().numpy()
            table[f"effective_gate_layer{layer_index}"] = scale.cpu().numpy() - 1.0

    return pd.DataFrame(table)


def node_scores(model, data):
    edges = edge_table(model, data)
    num_nodes = data.num_nodes
    weight_column = "edge_scale_layer1" if "edge_scale_layer1" in edges.columns else "pheromone"
    incoming = edges.groupby("target")[weight_column].sum().reindex(range(num_nodes), fill_value=0)
    outgoing = edges.groupby("source")[weight_column].sum().reindex(range(num_nodes), fill_value=0)
    return pd.DataFrame({
        "node": range(num_nodes),
        "memory_flow_in": incoming.values,
        "memory_flow_out": outgoing.values,
        "memory_flow_total": incoming.values + outgoing.values,
    })


def centrality_analysis(model, data, approximate_betweenness=250):
    edges = edge_table(model, data)
    weight_column = "edge_scale_layer1" if "edge_scale_layer1" in edges.columns else "pheromone"
    graph = nx.DiGraph()
    graph.add_nodes_from(range(data.num_nodes))
    graph.add_weighted_edges_from(edges[["source", "target", weight_column]].itertuples(index=False, name=None))
    undirected = graph.to_undirected()

    degree = dict(undirected.degree())
    pagerank = nx.pagerank(graph, weight=None)
    weighted_pagerank = nx.pagerank(graph, weight="weight")
    k = min(approximate_betweenness, data.num_nodes)
    betweenness = nx.betweenness_centrality(undirected, k=k, seed=0)
    closeness = nx.closeness_centrality(undirected)
    try:
        eigenvector = nx.eigenvector_centrality(undirected, max_iter=2000)
    except nx.PowerIterationFailedConvergence:
        eigenvector = {node: np.nan for node in undirected.nodes()}

    scores = node_scores(model, data)
    centralities = {
        "degree": degree, "pagerank": pagerank, "weighted_pagerank": weighted_pagerank,
        "betweenness": betweenness, "closeness": closeness, "eigenvector": eigenvector,
    }
    for name, values in centralities.items():
        scores[name] = scores["node"].map(values)

    rows = []
    for name in centralities:
        valid = scores[["memory_flow_total", name]].dropna()
        if len(valid) < 2 or valid["memory_flow_total"].nunique() < 2 or valid[name].nunique() < 2:
            pearson_r = pearson_p = np.nan
            spearman_rho = spearman_p = np.nan
        else:
            pearson_r, pearson_p = pearsonr(valid["memory_flow_total"], valid[name])
            spearman_rho, spearman_p = spearmanr(valid["memory_flow_total"], valid[name])
        rows.append({
            "centrality": name, "pearson_r": pearson_r, "pearson_p": pearson_p,
            "spearman_rho": spearman_rho, "spearman_p": spearman_p,
        })
    return scores, pd.DataFrame(rows)
