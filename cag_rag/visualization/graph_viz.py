"""
Graph Visualization  —  G' = (V, E ∪ C)
-----------------------------------------
Produces publication-ready plots of the augmented knowledge graph,
highlighting conflict edges (red dashed) vs. relation edges (gray).

Two main plots
--------------
1. plot_augmented_graph   — full G', truncated to max_nodes for readability
2. plot_conflict_subgraph — only nodes involved in at least one conflict edge

Usage
-----
    python -m cag_rag.visualization.graph_viz

Or from code:
    from cag_rag.visualization.graph_viz import plot_augmented_graph
    plot_augmented_graph(index, title="Super Bowl 50", output_path="figs/graph.png")
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import matplotlib
matplotlib.use("Agg")   # headless — works without a display
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import numpy as np

from ..graph.index import AugmentedGraphIndex, EDGE_TYPE_CONFLICT, EDGE_TYPE_RELATION

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------
_COL_NODE_NORMAL   = "#AEC6E8"   # light blue  — regular nodes
_COL_NODE_CONFLICT = "#FF8C69"   # salmon      — nodes in at least one conflict
_COL_EDGE_RELATION = "#999999"   # gray        — relation edges
_COL_EDGE_CONFLICT = "#D62728"   # red         — conflict edges


# ---------------------------------------------------------------------------
# plot_augmented_graph
# ---------------------------------------------------------------------------

def plot_augmented_graph(
    index: AugmentedGraphIndex,
    title: str = "Augmented Knowledge Graph G'",
    max_nodes: int = 40,
    output_path: str = "figs/augmented_graph.png",
    figsize: tuple = (14, 10),
    seed: int = 42,
) -> str:
    """
    Draw G' with relation edges (gray solid) and conflict edges (red dashed).

    Parameters
    ----------
    index        : AugmentedGraphIndex
    title        : plot title
    max_nodes    : cap node count for legibility (highest-degree nodes kept)
    output_path  : where to save the PNG
    figsize      : matplotlib figure size
    seed         : layout random seed

    Returns
    -------
    output_path  : the saved file path
    """
    G = index.G
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    # --- Select nodes to display ---
    if G.number_of_nodes() > max_nodes:
        # Keep the max_nodes nodes with highest total degree
        top_nodes = sorted(G.nodes(), key=lambda n: G.degree(n), reverse=True)[:max_nodes]
        subG = G.subgraph(top_nodes).copy()
        truncated = True
    else:
        subG = G
        truncated = False

    # --- Classify nodes ---
    conflict_nodes = set()
    for u, v, data in subG.edges(data=True):
        if data.get("edge_type") == EDGE_TYPE_CONFLICT:
            conflict_nodes.add(u)
            conflict_nodes.add(v)

    node_colors = [
        _COL_NODE_CONFLICT if n in conflict_nodes else _COL_NODE_NORMAL
        for n in subG.nodes()
    ]

    # --- Classify edges ---
    relation_edges = [
        (u, v) for u, v, d in subG.edges(data=True)
        if d.get("edge_type") == EDGE_TYPE_RELATION
    ]
    conflict_edges = [
        (u, v) for u, v, d in subG.edges(data=True)
        if d.get("edge_type") == EDGE_TYPE_CONFLICT
    ]

    # --- Layout ---
    pos = nx.spring_layout(subG, seed=seed, k=1.5 / max(1, np.sqrt(subG.number_of_nodes())))

    # --- Node labels (truncated) ---
    labels = {
        n: (subG.nodes[n].get("label", str(n))[:18] + "…"
            if len(subG.nodes[n].get("label", str(n))) > 18
            else subG.nodes[n].get("label", str(n)))
        for n in subG.nodes()
    }

    # --- Draw ---
    fig, ax = plt.subplots(figsize=figsize)

    nx.draw_networkx_nodes(subG, pos, node_color=node_colors,
                           node_size=600, alpha=0.9, ax=ax)
    nx.draw_networkx_labels(subG, pos, labels=labels,
                            font_size=7, font_color="#222222", ax=ax)

    if relation_edges:
        nx.draw_networkx_edges(subG, pos, edgelist=relation_edges,
                               edge_color=_COL_EDGE_RELATION, arrows=True,
                               arrowsize=12, width=1.0, alpha=0.6,
                               connectionstyle="arc3,rad=0.05", ax=ax)

    if conflict_edges:
        nx.draw_networkx_edges(subG, pos, edgelist=conflict_edges,
                               edge_color=_COL_EDGE_CONFLICT, arrows=True,
                               arrowsize=16, width=2.0, alpha=0.85,
                               style="dashed",
                               connectionstyle="arc3,rad=0.2", ax=ax)

    # --- Edge predicate labels (relation edges only, to avoid clutter) ---
    rel_edge_labels = {}
    for u, v, data in subG.edges(data=True):
        if data.get("edge_type") == EDGE_TYPE_RELATION:
            pred = data.get("predicate", "")[:12]
            rel_edge_labels[(u, v)] = pred
    if rel_edge_labels:
        nx.draw_networkx_edge_labels(subG, pos, edge_labels=rel_edge_labels,
                                     font_size=6, alpha=0.7, ax=ax)

    # --- Legend ---
    legend_handles = [
        mpatches.Patch(color=_COL_NODE_NORMAL,   label="Regular node"),
        mpatches.Patch(color=_COL_NODE_CONFLICT, label=f"Conflict node ({len(conflict_nodes)})"),
        mpatches.Patch(color=_COL_EDGE_RELATION, label=f"Relation edge ({len(relation_edges)})"),
        mpatches.Patch(color=_COL_EDGE_CONFLICT, label=f"Conflict edge ({len(conflict_edges)})"),
    ]
    ax.legend(handles=legend_handles, loc="upper left", fontsize=9,
              framealpha=0.9)

    suffix = f"  (top {max_nodes} nodes shown)" if truncated else ""
    stats = (f"  |V|={subG.number_of_nodes()}  "
             f"|E_rel|={len(relation_edges)}  "
             f"|E_con|={len(conflict_edges)}")
    ax.set_title(f"{title}{suffix}\n{stats}", fontsize=12, pad=12)
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved graph plot → {output_path}")
    return output_path


# ---------------------------------------------------------------------------
# plot_conflict_subgraph
# ---------------------------------------------------------------------------

def plot_conflict_subgraph(
    index: AugmentedGraphIndex,
    title: str = "Conflict Neighbourhood in G'",
    output_path: str = "figs/conflict_subgraph.png",
    figsize: tuple = (12, 8),
    seed: int = 0,
) -> str:
    """
    Draw only the nodes and edges involved in conflicts.
    Includes the 1-hop relation neighbourhood of each conflict node
    for context.

    Returns
    -------
    output_path
    """
    G = index.G
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    # Find all conflict-edge endpoints
    conflict_nodes: set = set()
    conflict_edges = []
    for u, v, data in G.edges(data=True):
        if data.get("edge_type") == EDGE_TYPE_CONFLICT:
            conflict_nodes.add(u)
            conflict_nodes.add(v)
            conflict_edges.append((u, v, data))

    if not conflict_nodes:
        logger.warning("No conflict edges found — nothing to plot.")
        return output_path

    # Add 1-hop relation neighbours for context
    context_nodes: set = set(conflict_nodes)
    for n in list(conflict_nodes):
        for nbr in list(G.predecessors(n)) + list(G.successors(n)):
            if G.number_of_nodes() <= 60 or len(context_nodes) < 60:
                context_nodes.add(nbr)

    subG = G.subgraph(context_nodes).copy()

    rel_edges = [
        (u, v) for u, v, d in subG.edges(data=True)
        if d.get("edge_type") == EDGE_TYPE_RELATION
    ]
    con_edges = [
        (u, v) for u, v, d in subG.edges(data=True)
        if d.get("edge_type") == EDGE_TYPE_CONFLICT
    ]

    node_colors = [
        _COL_NODE_CONFLICT if n in conflict_nodes else _COL_NODE_NORMAL
        for n in subG.nodes()
    ]

    pos = nx.spring_layout(subG, seed=seed, k=2.0 / max(1, np.sqrt(subG.number_of_nodes())))

    labels = {
        n: (subG.nodes[n].get("label", str(n))[:20] + "…"
            if len(subG.nodes[n].get("label", str(n))) > 20
            else subG.nodes[n].get("label", str(n)))
        for n in subG.nodes()
    }

    fig, ax = plt.subplots(figsize=figsize)

    nx.draw_networkx_nodes(subG, pos, node_color=node_colors,
                           node_size=700, alpha=0.9, ax=ax)
    nx.draw_networkx_labels(subG, pos, labels=labels,
                            font_size=8, ax=ax)

    if rel_edges:
        nx.draw_networkx_edges(subG, pos, edgelist=rel_edges,
                               edge_color=_COL_EDGE_RELATION, arrows=True,
                               arrowsize=14, width=1.2, alpha=0.5,
                               connectionstyle="arc3,rad=0.05", ax=ax)

    if con_edges:
        nx.draw_networkx_edges(subG, pos, edgelist=con_edges,
                               edge_color=_COL_EDGE_CONFLICT, arrows=True,
                               arrowsize=18, width=2.5, alpha=0.9,
                               style="dashed",
                               connectionstyle="arc3,rad=0.25", ax=ax)

    # Conflict edge type annotations
    con_labels = {}
    for u, v, d in subG.edges(data=True):
        if d.get("edge_type") == EDGE_TYPE_CONFLICT:
            ctype = d.get("conflict_type", "conflict")[:10]
            con_labels[(u, v)] = ctype
    if con_labels:
        nx.draw_networkx_edge_labels(subG, pos, edge_labels=con_labels,
                                     font_size=7, font_color=_COL_EDGE_CONFLICT,
                                     ax=ax)

    legend_handles = [
        mpatches.Patch(color=_COL_NODE_NORMAL,   label="Context node"),
        mpatches.Patch(color=_COL_NODE_CONFLICT, label=f"Conflict node ({len(conflict_nodes)})"),
        mpatches.Patch(color=_COL_EDGE_RELATION, label=f"Relation edge ({len(rel_edges)})"),
        mpatches.Patch(color=_COL_EDGE_CONFLICT, label=f"Conflict edge ({len(con_edges)})"),
    ]
    ax.legend(handles=legend_handles, loc="upper left", fontsize=9, framealpha=0.9)
    ax.set_title(f"{title}\n|conflict nodes|={len(conflict_nodes)}  "
                 f"|conflict edges|={len(con_edges)}", fontsize=12, pad=12)
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved conflict subgraph → {output_path}")
    return output_path


# ---------------------------------------------------------------------------
# Demo entry point  —  uses hardcoded triples from test_pipeline
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    logging.basicConfig(level=logging.INFO)

    from cag_rag.conflict.semantic_filter import Triple
    from cag_rag.conflict.detector import ConflictDetector
    from cag_rag.graph.builder import KnowledgeGraphBuilder, Document
    from cag_rag.graph.index import AugmentedGraphIndex
    from cag_rag.injection.inject_conflicts import ConflictInjector

    docs = [
        Document("d1", "Albert Einstein was born in Ulm in 1879. He developed the theory of relativity."),
        Document("d2", "Einstein was born in Munich. He won the Nobel Prize in Physics in 1921."),
        Document("d3", "Marie Curie was born in Warsaw in 1867. She discovered polonium and radium."),
        Document("d4", "Curie won the Nobel Prize twice. She was the first woman to win a Nobel Prize."),
    ]

    builder = KnowledgeGraphBuilder(backend="spacy")
    _, triples = builder.build(docs)
    print(f"Extracted {len(triples)} triples")

    injector = ConflictInjector(injection_rate=0.4, seed=42)
    inj_result = injector.inject(triples)

    detector = ConflictDetector(
        nli_model_name="cross-encoder/nli-deberta-v3-base",
        contradiction_threshold=0.35,
    )
    conflict_edges = detector.detect(inj_result.augmented_triples)
    print(f"Detected {len(conflict_edges)} conflict edges")

    base_graph = KnowledgeGraphBuilder._triples_to_graph(inj_result.augmented_triples)
    index = AugmentedGraphIndex.from_base_graph(base_graph, conflict_edges)

    p1 = plot_augmented_graph(index, title="Demo — Augmented Graph G'",
                              output_path="figs/augmented_graph.png")
    p2 = plot_conflict_subgraph(index, title="Demo — Conflict Subgraph",
                                output_path="figs/conflict_subgraph.png")
    print(f"Saved: {p1}")
    print(f"Saved: {p2}")
