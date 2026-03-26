"""
Augmented Graph Index  G' = (V, E ∪ C)
----------------------------------------
Takes a base knowledge graph G = (V, E) and a list of ConflictEdge objects,
and produces the heterogeneous augmented index G' by inserting conflict edges
as a new edge type ("conflict") alongside the original relation edges.

Key design decisions
--------------------
* Relation edges  → edge_type = "relation"
* Conflict edges  → edge_type = "conflict"
  These are stored as *multi-edges* (using nx.MultiDiGraph) so that two nodes
  can be connected by both a relation edge and one or more conflict edges.

* Each conflict edge carries the full ConflictEdge metadata:
    - conflicting_predicate
    - source_a, source_b
    - detection_confidence
    - conflict_type  (attribute | temporal | logical)
    - edge_id        (UUID from the detector)

* The index exposes three retrieval methods used by the conflict-aware retriever:
    get_conflict_edges(node_id)     → all conflict edges incident to a node
    get_conflict_neighbors(node_id) → nodes on the other end of conflict edges
    get_subgraph(node_ids)          → induced subgraph over a set of nodes

Usage
-----
    from cag_rag.graph.builder import KnowledgeGraphBuilder, Document
    from cag_rag.conflict.detector import ConflictDetector
    from cag_rag.graph.index import AugmentedGraphIndex

    builder  = KnowledgeGraphBuilder()
    G, triples = builder.build(documents)

    detector = ConflictDetector()
    conflict_edges = detector.detect(triples)

    index = AugmentedGraphIndex.from_base_graph(G, conflict_edges)

    # Retrieve conflict neighbourhood for a node
    neighbors = index.get_conflict_neighbors("albert_einstein")
"""

from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import networkx as nx

from ..conflict.detector import ConflictEdge
from .builder import _entity_id

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Edge key constants
# ---------------------------------------------------------------------------

EDGE_TYPE_RELATION = "relation"
EDGE_TYPE_CONFLICT  = "conflict"


# ---------------------------------------------------------------------------
# AugmentedGraphIndex
# ---------------------------------------------------------------------------

class AugmentedGraphIndex:
    """
    Heterogeneous augmented graph index G' = (V, E ∪ C).

    Internally uses nx.MultiDiGraph to allow multiple edge types
    between the same node pair (e.g., both a "relation" and a "conflict" edge).

    Attributes
    ----------
    G : nx.MultiDiGraph
        The full augmented graph.
    conflict_edge_index : Dict[str, List[Dict]]
        Maps node_id → list of conflict edge attribute dicts incident to that node.
        Pre-built for O(1) conflict neighbourhood lookup.
    stats : Dict
        Summary statistics populated at build time.
    """

    def __init__(self, G: nx.MultiDiGraph):
        self.G = G
        self.conflict_edge_index: Dict[str, List[Dict]] = {}
        self.stats: Dict = {}
        self._build_conflict_index()
        self._compute_stats()

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_base_graph(
        cls,
        base_graph: nx.DiGraph,
        conflict_edges: List[ConflictEdge],
    ) -> "AugmentedGraphIndex":
        """
        Build G' from:
          - base_graph : the standard knowledge graph G = (V, E)
          - conflict_edges : ConflictEdge objects from ConflictDetector
        """
        # Convert DiGraph → MultiDiGraph (preserves all existing edges)
        G_prime = nx.MultiDiGraph(base_graph)

        # Insert conflict edges
        n_added = 0
        for ce in conflict_edges:
            subj_id = _entity_id(ce.triple_a.subject)
            obj_id  = _entity_id(ce.triple_b.subject)   # the other *entity*, not the object value

            # Ensure both entity nodes exist (they should, but guard against
            # cases where the detector was run on external triple lists)
            for node_id, triple in [(subj_id, ce.triple_a), (obj_id, ce.triple_b)]:
                if node_id not in G_prime:
                    G_prime.add_node(
                        node_id,
                        label=triple.subject,
                        source_docs={triple.source_doc},
                        entity_type="",
                    )
                else:
                    G_prime.nodes[node_id].setdefault("source_docs", set()).add(triple.source_doc)

            # Add the conflict edge (undirected semantics — add both directions)
            edge_attrs = cls._conflict_edge_attrs(ce)
            G_prime.add_edge(subj_id, obj_id, **edge_attrs)
            G_prime.add_edge(obj_id, subj_id, **edge_attrs)   # symmetric conflict
            n_added += 1

        logger.info(
            f"Augmented index built: {G_prime.number_of_nodes()} nodes, "
            f"{G_prime.number_of_edges()} edges ({n_added} conflict edges injected)"
        )
        return cls(G_prime)

    @classmethod
    def from_triples_and_conflicts(
        cls,
        triples,           # List[Triple]
        conflict_edges: List[ConflictEdge],
    ) -> "AugmentedGraphIndex":
        """Convenience: build base graph from triples then augment."""
        from .builder import KnowledgeGraphBuilder
        base_graph = KnowledgeGraphBuilder.build_from_triples(triples)   # static call
        # build_from_triples is a static method — call via the class
        base_graph = KnowledgeGraphBuilder._triples_to_graph(triples)
        return cls.from_base_graph(base_graph, conflict_edges)

    # ------------------------------------------------------------------
    # Retrieval API  (used by conflict_aware retriever)
    # ------------------------------------------------------------------

    def get_conflict_edges(self, node_id: str) -> List[Dict]:
        """
        Return all conflict edge attribute dicts incident to node_id.
        Returns [] if the node has no conflict edges.
        """
        node_id = _entity_id(node_id)
        return self.conflict_edge_index.get(node_id, [])

    def get_conflict_neighbors(self, node_id: str) -> List[str]:
        """
        Return node IDs of all nodes connected to node_id via a conflict edge.
        """
        edges = self.get_conflict_edges(node_id)
        neighbors = []
        node_id_norm = _entity_id(node_id)
        for e in edges:
            other = e["target"] if e["source"] == node_id_norm else e["source"]
            neighbors.append(other)
        return list(set(neighbors))

    def get_subgraph(self, node_ids: List[str]) -> nx.MultiDiGraph:
        """
        Return the induced subgraph (all edges between the given nodes).
        Includes both relation and conflict edges.
        """
        norm_ids = [_entity_id(n) for n in node_ids]
        existing = [n for n in norm_ids if n in self.G]
        return self.G.subgraph(existing).copy()

    def get_relation_subgraph(self, node_ids: List[str]) -> nx.MultiDiGraph:
        """Induced subgraph containing only relation edges (no conflict edges)."""
        sub = self.get_subgraph(node_ids)
        edges_to_remove = [
            (u, v, k)
            for u, v, k, d in sub.edges(data=True, keys=True)
            if d.get("edge_type") == EDGE_TYPE_CONFLICT
        ]
        sub.remove_edges_from(edges_to_remove)
        return sub

    def get_conflict_subgraph(self, node_ids: List[str]) -> nx.MultiDiGraph:
        """Induced subgraph containing only conflict edges."""
        sub = self.get_subgraph(node_ids)
        edges_to_remove = [
            (u, v, k)
            for u, v, k, d in sub.edges(data=True, keys=True)
            if d.get("edge_type") != EDGE_TYPE_CONFLICT
        ]
        sub.remove_edges_from(edges_to_remove)
        return sub

    def has_conflict(self, node_id_a: str, node_id_b: str) -> bool:
        """Return True if there is at least one conflict edge between two nodes."""
        a, b = _entity_id(node_id_a), _entity_id(node_id_b)
        if not (self.G.has_node(a) and self.G.has_node(b)):
            return False
        return any(
            d.get("edge_type") == EDGE_TYPE_CONFLICT
            for _, _, d in self.G.edges(a, data=True)
            if _ == a  # out-edges from a
        )

    def node_label(self, node_id: str) -> str:
        """Return the surface-form label of a node."""
        node_id = _entity_id(node_id)
        return self.G.nodes[node_id].get("label", node_id) if self.G.has_node(node_id) else node_id

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def save(self, path: str):
        """Persist the augmented index to disk (pickle)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.info(f"Index saved to {path}")

    @staticmethod
    def load(path: str) -> "AugmentedGraphIndex":
        """Load a previously saved augmented index."""
        with open(path, "rb") as f:
            obj = pickle.load(f)
        logger.info(f"Index loaded from {path}")
        return obj

    def export_conflict_edges_json(self, path: str):
        """Export all conflict edges as JSON (for inspection / debugging)."""
        records = []
        seen = set()
        for edges in self.conflict_edge_index.values():
            for e in edges:
                eid = e.get("edge_id", "")
                if eid not in seen:
                    seen.add(eid)
                    records.append(e)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(records, f, indent=2, default=str)
        logger.info(f"Exported {len(records)} conflict edges to {path}")

    # ------------------------------------------------------------------
    # Summary / diagnostics
    # ------------------------------------------------------------------

    def summary(self) -> str:
        s = self.stats
        lines = [
            "AugmentedGraphIndex summary",
            f"  nodes            : {s.get('n_nodes', '?')}",
            f"  relation edges   : {s.get('n_relation_edges', '?')}",
            f"  conflict edges   : {s.get('n_conflict_edges', '?')}",
            f"  nodes w/ conflict: {s.get('n_conflict_nodes', '?')}",
            f"  conflict types   : {s.get('conflict_type_counts', {})}",
        ]
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _conflict_edge_attrs(ce: ConflictEdge) -> Dict:
        subj_id = _entity_id(ce.triple_a.subject)
        obj_id  = _entity_id(ce.triple_b.subject)
        return {
            "edge_type":              EDGE_TYPE_CONFLICT,
            "edge_id":                ce.edge_id,
            "conflicting_predicate":  ce.conflicting_predicate,
            "conflict_type":          ce.conflict_type,
            "detection_confidence":   ce.detection_confidence,
            "source_a":               ce.source_a,
            "source_b":               ce.source_b,
            "object_a":               ce.triple_a.obj,
            "object_b":               ce.triple_b.obj,
            "subject_sim":            ce.subject_sim,
            "predicate_sim":          ce.predicate_sim,
            "object_sim":             ce.object_sim,
            "source":                 subj_id,
            "target":                 obj_id,
        }

    def _build_conflict_index(self):
        """Pre-build node → conflict edges lookup for O(1) retrieval."""
        index: Dict[str, List[Dict]] = {}
        for u, v, data in self.G.edges(data=True):
            if data.get("edge_type") != EDGE_TYPE_CONFLICT:
                continue
            index.setdefault(u, []).append(data)
            index.setdefault(v, []).append(data)
        self.conflict_edge_index = index

    def _compute_stats(self):
        n_rel = sum(
            1 for _, _, d in self.G.edges(data=True)
            if d.get("edge_type") == EDGE_TYPE_RELATION
        )
        n_conf = sum(
            1 for _, _, d in self.G.edges(data=True)
            if d.get("edge_type") == EDGE_TYPE_CONFLICT
        )
        type_counts: Dict[str, int] = {}
        for _, _, d in self.G.edges(data=True):
            if d.get("edge_type") == EDGE_TYPE_CONFLICT:
                ct = d.get("conflict_type", "unknown")
                type_counts[ct] = type_counts.get(ct, 0) + 1

        self.stats = {
            "n_nodes":            self.G.number_of_nodes(),
            "n_relation_edges":   n_rel,
            "n_conflict_edges":   n_conf,
            "n_conflict_nodes":   len(self.conflict_edge_index),
            "conflict_type_counts": type_counts,
        }
