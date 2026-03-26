"""
Conflict-Aware Retrieval
-------------------------
Given a query, retrieves:
  1. Relevant subgraph  — standard BFS/embedding-based retrieval from G'
  2. Conflict neighbourhood — for every retrieved node that has conflict edges,
     the contradicting nodes and their conflict edge metadata are also included

The retriever returns a RetrievalResult that the generator uses to build
a structured prompt containing both the factual subgraph and any detected
disagreements.

Retrieval strategy (two stages)
--------------------------------
Stage A — Seed node identification:
  Encode the query with a SentenceTransformer, compute cosine similarity
  against pre-computed node label embeddings, and take the top-k matches
  as seed nodes.

Stage B — Subgraph expansion:
  BFS from each seed node up to `depth` hops using only "relation" edges.
  Then expand each visited node's conflict neighbourhood.

Output
------
  RetrievalResult
    .subgraph_nodes      : List[str]   — node IDs in the retrieved subgraph
    .subgraph_triples    : List[Dict]  — (subject, predicate, object, source)
    .conflict_contexts   : List[ConflictContext]
        .node_id             : str
        .conflicting_node_id : str
        .conflict_edge       : Dict    — full conflict edge attributes
        .node_a_text         : str     — "subject predicate object_a (src: ...)"
        .node_b_text         : str     — "subject predicate object_b (src: ...)"
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from ..graph.index import AugmentedGraphIndex, EDGE_TYPE_CONFLICT, EDGE_TYPE_RELATION

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Output data structures
# ---------------------------------------------------------------------------

@dataclass
class ConflictContext:
    """Structured representation of one conflict edge for the generator."""
    node_id: str                 # node in the retrieved subgraph
    conflicting_node_id: str     # the node it conflicts with
    conflict_edge: Dict          # raw edge attribute dict from G'
    node_a_text: str             # human-readable fact from source A
    node_b_text: str             # human-readable fact from source B

    @property
    def predicate(self) -> str:
        return self.conflict_edge.get("conflicting_predicate", "?")

    @property
    def confidence(self) -> float:
        return self.conflict_edge.get("detection_confidence", 0.0)

    @property
    def conflict_type(self) -> str:
        return self.conflict_edge.get("conflict_type", "unknown")

    def to_prompt_text(self) -> str:
        """Render as a structured text block for the LLM prompt."""
        return (
            f"[CONFLICT — {self.conflict_type}] "
            f"Predicate: '{self.predicate}'\n"
            f"  Source A ({self.conflict_edge.get('source_a', '?')}): {self.node_a_text}\n"
            f"  Source B ({self.conflict_edge.get('source_b', '?')}): {self.node_b_text}\n"
            f"  Confidence: {self.confidence:.2f}"
        )


@dataclass
class RetrievalResult:
    """All context returned by the conflict-aware retriever for one query."""
    query: str
    seed_nodes: List[str]                           = field(default_factory=list)
    subgraph_nodes: List[str]                       = field(default_factory=list)
    subgraph_triples: List[Dict]                    = field(default_factory=list)
    conflict_contexts: List[ConflictContext]        = field(default_factory=list)

    @property
    def has_conflicts(self) -> bool:
        return len(self.conflict_contexts) > 0

    def to_context_string(self) -> str:
        """
        Render the retrieval result as a plain-text context block
        suitable for insertion into an LLM prompt.
        """
        lines = ["=== Retrieved Knowledge ==="]

        if self.subgraph_triples:
            lines.append("\n--- Facts ---")
            for t in self.subgraph_triples:
                lines.append(
                    f"  {t['subject']} —[{t['predicate']}]→ {t['object']} "
                    f"(src: {t.get('source_doc', '?')})"
                )

        if self.conflict_contexts:
            lines.append("\n--- Detected Conflicts ---")
            for cc in self.conflict_contexts:
                lines.append(cc.to_prompt_text())
        else:
            lines.append("\n[No conflicts detected in retrieved context]")

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Embedding helper (shared with conflict detector for reuse)
# ---------------------------------------------------------------------------

class _NodeEmbeddingCache:
    """Lazily computes and caches node label embeddings for a given index."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self._model_name = model_name
        self._model = None
        self._embeddings: Optional[np.ndarray] = None
        self._node_ids: Optional[List[str]] = None

    def _load_model(self):
        from sentence_transformers import SentenceTransformer
        self._model = SentenceTransformer(self._model_name)

    def build(self, index: AugmentedGraphIndex):
        if self._model is None:
            self._load_model()
        self._node_ids = list(index.G.nodes())
        labels = [index.G.nodes[n].get("label", n) for n in self._node_ids]
        logger.info(f"Computing embeddings for {len(labels)} nodes …")
        self._embeddings = self._model.encode(labels, normalize_embeddings=True,
                                              show_progress_bar=False)

    def top_k(self, query: str, k: int) -> List[Tuple[str, float]]:
        """Return (node_id, similarity) for the top-k most similar nodes."""
        if self._model is None:
            self._load_model()
        q_emb = self._model.encode([query], normalize_embeddings=True)[0]  # (D,)
        sims = self._embeddings @ q_emb                                      # (N,)
        top_indices = np.argsort(sims)[::-1][:k]
        return [(self._node_ids[i], float(sims[i])) for i in top_indices]


# ---------------------------------------------------------------------------
# ConflictAwareRetriever
# ---------------------------------------------------------------------------

class ConflictAwareRetriever:
    """
    Retrieves relevant subgraphs from G' together with their conflict neighbourhoods.

    Parameters
    ----------
    index : AugmentedGraphIndex
        The augmented graph G' = (V, E ∪ C).
    top_k_seeds : int
        Number of seed nodes selected by embedding similarity to the query.
    bfs_depth : int
        BFS expansion depth along relation edges from seed nodes.
    min_conflict_confidence : float
        Only include conflict contexts above this confidence threshold.
    embedding_model : str
        SentenceTransformer model for query → node similarity.
    """

    def __init__(
        self,
        index: AugmentedGraphIndex,
        top_k_seeds: int = 5,
        bfs_depth: int = 2,
        min_conflict_confidence: float = 0.4,
        embedding_model: str = "all-MiniLM-L6-v2",
    ):
        self.index = index
        self.top_k_seeds = top_k_seeds
        self.bfs_depth = bfs_depth
        self.min_conflict_confidence = min_conflict_confidence
        self._emb_cache = _NodeEmbeddingCache(embedding_model)
        self._emb_cache.build(index)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def retrieve(self, query: str) -> RetrievalResult:
        """
        Full conflict-aware retrieval for a single query.
        Returns a RetrievalResult ready to be passed to the generator.
        """
        result = RetrievalResult(query=query)

        # --- Stage A: Seed nodes ---
        seed_candidates = self._emb_cache.top_k(query, self.top_k_seeds)
        result.seed_nodes = [node_id for node_id, _ in seed_candidates]
        logger.info(f"Seeds: {result.seed_nodes}")

        # --- Stage B: BFS subgraph expansion (relation edges only) ---
        visited = self._bfs_expand(result.seed_nodes, depth=self.bfs_depth)
        result.subgraph_nodes = list(visited)

        # --- Extract triples from the subgraph ---
        result.subgraph_triples = self._extract_triples(visited)

        # --- Conflict neighbourhood expansion ---
        result.conflict_contexts = self._expand_conflicts(visited)

        logger.info(
            f"Retrieved {len(result.subgraph_nodes)} nodes, "
            f"{len(result.subgraph_triples)} triples, "
            f"{len(result.conflict_contexts)} conflict contexts"
        )
        return result

    # ------------------------------------------------------------------
    # BFS expansion along relation edges
    # ------------------------------------------------------------------

    def _bfs_expand(self, seeds: List[str], depth: int) -> Set[str]:
        visited: Set[str] = set()
        queue: deque[Tuple[str, int]] = deque()

        for s in seeds:
            if s in self.index.G:
                queue.append((s, 0))
                visited.add(s)

        while queue:
            node, d = queue.popleft()
            if d >= depth:
                continue
            for _, neighbor, data in self.index.G.out_edges(node, data=True):
                if data.get("edge_type") != EDGE_TYPE_RELATION:
                    continue
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, d + 1))

        return visited

    # ------------------------------------------------------------------
    # Triple extraction
    # ------------------------------------------------------------------

    def _extract_triples(self, node_ids: Set[str]) -> List[Dict]:
        triples = []
        G = self.index.G
        for u in node_ids:
            for _, v, data in G.out_edges(u, data=True):
                if data.get("edge_type") != EDGE_TYPE_RELATION:
                    continue
                if v not in node_ids:
                    continue
                triples.append({
                    "subject":   G.nodes[u].get("label", u),
                    "predicate": data.get("predicate", "related_to"),
                    "object":    G.nodes[v].get("label", v),
                    "source_doc": data.get("source_doc", ""),
                    "triple_id":  data.get("triple_id", ""),
                })
        return triples

    # ------------------------------------------------------------------
    # Conflict neighbourhood expansion
    # ------------------------------------------------------------------

    def _expand_conflicts(self, subgraph_nodes: Set[str]) -> List[ConflictContext]:
        contexts: List[ConflictContext] = []
        seen_edge_ids: Set[str] = set()
        G = self.index.G

        for node_id in subgraph_nodes:
            conflict_edges = self.index.get_conflict_edges(node_id)
            for ce in conflict_edges:
                edge_id = ce.get("edge_id", "")
                if edge_id in seen_edge_ids:
                    continue
                if ce.get("detection_confidence", 0.0) < self.min_conflict_confidence:
                    continue
                seen_edge_ids.add(edge_id)

                src = ce.get("source", node_id)
                tgt = ce.get("target", "")
                conflicting_node = tgt if src == node_id else src

                # Build human-readable fact strings
                subj_label = G.nodes[src].get("label", src) if G.has_node(src) else src
                predicate   = ce.get("conflicting_predicate", "?")
                obj_a       = ce.get("object_a", "?")
                obj_b       = ce.get("object_b", "?")
                src_a       = ce.get("source_a", "?")
                src_b       = ce.get("source_b", "?")

                contexts.append(ConflictContext(
                    node_id=node_id,
                    conflicting_node_id=conflicting_node,
                    conflict_edge=ce,
                    node_a_text=f"{subj_label} {predicate} {obj_a}",
                    node_b_text=f"{subj_label} {predicate} {obj_b}",
                ))

        return contexts
