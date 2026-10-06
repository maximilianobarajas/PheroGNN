from __future__ import annotations
import numpy as np
import pandas as pd
import networkx as nx
from scipy.stats import pearsonr, spearmanr
from .models import PheroGNN, PheroGNNv7, PheroAPPNP, PheroSAGE, PheroSAGEAPPNP, PheroGCNII

PHERO_MODELS = (PheroGNN, PheroGNNv7, PheroAPPNP, PheroSAGE, PheroSAGEAPPNP, PheroGCNII)


def edge_table(model, data):
    if not hasattr(model, "effective_tau"):
        raise TypeError(
            "Interpretability requires a model with a single pheromone trail "
            "(effective_tau()); an ensemble of several has no single trail to report."
        )

    src, dst = data.edge_index.detach().cpu().numpy()
    tau = model.effective_tau().detach().cpu().numpy()

    return pd.DataFrame({"source": src, "target": dst, "pheromone": tau})


def node_scores(model, data):
    edges = edge_table(model, data)
    num_nodes = data.num_nodes
    incoming = edges.groupby("target")["pheromone"].sum().reindex(range(num_nodes), fill_value=0)
    outgoing = edges.groupby("source")["pheromone"].sum().reindex(range(num_nodes), fill_value=0)
    return pd.DataFrame({
        "node": range(num_nodes),
        "pheromone_in": incoming.values,
        "pheromone_out": outgoing.values,
        "pheromone_total": incoming.values + outgoing.values,
    })


def centrality_analysis(model, data, approximate_betweenness=250):
    edges = edge_table(model, data)
    graph = nx.DiGraph()
    graph.add_nodes_from(range(data.num_nodes))
    graph.add_weighted_edges_from(edges[["source", "target", "pheromone"]].itertuples(index=False, name=None))
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
        valid = scores[["pheromone_total", name]].dropna()
        if len(valid) < 2 or valid["pheromone_total"].nunique() < 2 or valid[name].nunique() < 2:
            pearson_r = pearson_p = np.nan
            spearman_rho = spearman_p = np.nan
        else:
            pearson_r, pearson_p = pearsonr(valid["pheromone_total"], valid[name])
            spearman_rho, spearman_p = spearmanr(valid["pheromone_total"], valid[name])
        rows.append({
            "centrality": name, "pearson_r": pearson_r, "pearson_p": pearson_p,
            "spearman_rho": spearman_rho, "spearman_p": spearman_p,
        })
    return scores, pd.DataFrame(rows)
