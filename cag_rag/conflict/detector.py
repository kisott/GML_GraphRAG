"""
Conflict Detection Module (Main Orchestrator)
----------------------------------------------
Combines SemanticFilter (Stage 1) and NLIClassifier (Stage 2) to produce
ConflictEdge objects that are inserted into the augmented graph index G'.

Pipeline per query / corpus:
  Triples → SemanticFilter → CandidatePairs → NLIClassifier → ConflictEdges

Each ConflictEdge carries:
  - The two conflicting triple IDs
  - The conflicting predicate
  - Source provenance for both sides
  - NLI contradiction probability (detection confidence)
  - Conflict type tag (attribute | temporal | logical)

Usage example
-------------
    from cag_rag.conflict.detector import ConflictDetector
    from cag_rag.conflict.semantic_filter import Triple

    triples = [
        Triple("Albert Einstein", "born_in", "Ulm", source_doc="wiki_v1"),
        Triple("Albert Einstein", "born_in", "Munich", source_doc="wiki_v2"),
        Triple("Marie Curie", "award", "Nobel Prize in Physics", source_doc="doc_a"),
        Triple("Marie Curie", "award", "Nobel Prize in Chemistry", source_doc="doc_b"),
    ]

    detector = ConflictDetector()
    edges = detector.detect(triples)
    for e in edges:
        print(e)
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from typing import List, Literal, Optional

from .semantic_filter import CandidatePair, SemanticFilter, Triple
from .nli_classifier import NLIClassifier

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# ConflictEdge — the unit inserted into G' = (V, E ∪ C)
# ---------------------------------------------------------------------------

ConflictType = Literal["attribute", "temporal", "logical", "unknown"]


@dataclass
class ConflictEdge:
    """
    Represents a detected conflict between two triples in the knowledge graph.
    Stored as a typed edge in the heterogeneous graph index.
    """
    edge_id: str                        # unique identifier for this conflict edge
    triple_a: Triple                    # first conflicting triple
    triple_b: Triple                    # second conflicting triple
    conflicting_predicate: str          # the shared predicate that defines the conflict axis
    source_a: str                       # provenance of triple_a
    source_b: str                       # provenance of triple_b
    detection_confidence: float         # NLI contradiction probability ∈ [0, 1]
    conflict_type: ConflictType = "unknown"
    subject_sim: float = 0.0            # similarity scores from Stage 1
    predicate_sim: float = 0.0
    object_sim: float = 0.0

    # Human-readable summary (populated automatically)
    summary: str = field(init=False)

    def __post_init__(self):
        self.summary = (
            f"[{self.conflict_type}] "
            f"'{self.triple_a.subject}' — {self.conflicting_predicate} — "
            f"'{self.triple_a.obj}' (src: {self.source_a}) "
            f"vs '{self.triple_b.obj}' (src: {self.source_b}) "
            f"[conf={self.detection_confidence:.3f}]"
        )

    def __repr__(self) -> str:
        return self.summary


# ---------------------------------------------------------------------------
# Conflict Type Heuristics
# ---------------------------------------------------------------------------

_TEMPORAL_KEYWORDS = {
    "year", "date", "born", "died", "founded", "established",
    "since", "until", "start", "end", "period", "era", "age",
    "published", "released", "appointed", "elected",
}

_NUMERICAL_KEYWORDS = {
    "population", "area", "height", "length", "weight", "distance",
    "number", "count", "size", "capacity", "speed", "temperature",
    "price", "cost", "revenue", "gdp",
}


def _infer_conflict_type(predicate: str, obj_a: str, obj_b: str) -> ConflictType:
    """
    Simple heuristic to tag a conflict as temporal, attribute, or logical.
    Can be replaced with a trained classifier later.
    """
    pred_lower = predicate.lower()

    # Temporal conflict: predicate contains a time-related keyword
    if any(kw in pred_lower for kw in _TEMPORAL_KEYWORDS):
        return "temporal"

    # Attribute conflict: one or both objects contain digits (numerical values)
    if any(ch.isdigit() for ch in obj_a + obj_b):
        if any(kw in pred_lower for kw in _NUMERICAL_KEYWORDS):
            return "attribute"

    # Logical conflict: catch-all for semantic contradictions
    return "logical"


# ---------------------------------------------------------------------------
# ConflictDetector
# ---------------------------------------------------------------------------

class ConflictDetector:
    """
    Two-stage conflict detection pipeline:
      Stage 1 — SemanticFilter  : narrows O(N²) triple pairs to candidates
      Stage 2 — NLIClassifier   : classifies each candidate as contradiction/neutral/entailment

    Parameters
    ----------
    subject_threshold : float
        Minimum cosine similarity for two subjects to be considered co-referent.
    predicate_threshold : float
        Minimum cosine similarity for two predicates to share the same axis.
    object_sim_max : float
        Maximum object similarity — pairs above this are paraphrases, not conflicts.
    contradiction_threshold : float
        Minimum NLI contradiction probability to emit a ConflictEdge.
    nli_model_name : str
        HuggingFace model identifier for the NLI cross-encoder.
    nli_mode : str
        "cross_encoder" (default, faster batch) or "pipeline" (simpler).
    embedding_model_name : str
        SentenceTransformer model for Stage 1 embeddings.
    device : str
        "cpu" or "cuda".
    use_soft_match : bool
        Whether to run embedding-based soft matching in addition to hard match.
    """

    def __init__(
        self,
        subject_threshold: float = 0.85,
        predicate_threshold: float = 0.80,
        object_sim_max: float = 0.90,
        contradiction_threshold: float = 0.50,
        nli_model_name: str = "cross-encoder/nli-deberta-v3-large",
        nli_mode: str = "cross_encoder",
        embedding_model_name: str = "all-MiniLM-L6-v2",
        device: str = "cpu",
        use_soft_match: bool = True,
    ):
        self.contradiction_threshold = contradiction_threshold

        self._semantic_filter = SemanticFilter(
            subject_threshold=subject_threshold,
            predicate_threshold=predicate_threshold,
            object_sim_max=object_sim_max,
            embedding_model_name=embedding_model_name,
            use_soft_match=use_soft_match,
        )
        self._nli = NLIClassifier(
            model_name=nli_model_name,
            mode=nli_mode,
            contradiction_threshold=contradiction_threshold,
            device=device,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self, triples: List[Triple]) -> List[ConflictEdge]:
        """
        Run the full two-stage pipeline on a list of triples.
        Returns ConflictEdge objects for all detected conflicts.
        """
        if len(triples) < 2:
            return []

        # --- Stage 1: Semantic filter ---
        logger.info(f"Stage 1: filtering {len(triples)} triples for candidates …")
        candidates: List[CandidatePair] = self._semantic_filter.filter(triples)
        logger.info(f"  → {len(candidates)} candidate pairs found")

        if not candidates:
            return []

        # --- Stage 2: NLI classification ---
        # Build (premise, hypothesis) pairs from the *full triple text*
        # so the model sees full context, not just the objects.
        pairs = [
            (c.triple_a.as_text(), c.triple_b.as_text())
            for c in candidates
        ]
        logger.info(f"Stage 2: running NLI on {len(pairs)} pairs …")
        nli_results = self._nli.classify_pairs(pairs)

        # --- Combine results into ConflictEdges ---
        edges: List[ConflictEdge] = []
        for candidate, nli in zip(candidates, nli_results):
            # Attach NLI result back to the candidate pair
            candidate.conflict_label = nli.label
            candidate.conflict_score = nli.contradiction_prob

            if nli.contradiction_prob >= self.contradiction_threshold:
                predicate = self._shared_predicate(candidate)
                ctype = _infer_conflict_type(
                    predicate, candidate.triple_a.obj, candidate.triple_b.obj
                )
                edge = ConflictEdge(
                    edge_id=str(uuid.uuid4()),
                    triple_a=candidate.triple_a,
                    triple_b=candidate.triple_b,
                    conflicting_predicate=predicate,
                    source_a=candidate.triple_a.source_doc,
                    source_b=candidate.triple_b.source_doc,
                    detection_confidence=nli.contradiction_prob,
                    conflict_type=ctype,
                    subject_sim=candidate.subject_sim,
                    predicate_sim=candidate.predicate_sim,
                    object_sim=candidate.object_sim,
                )
                edges.append(edge)
                logger.debug(f"  Conflict: {edge}")

        logger.info(f"Detected {len(edges)} conflict edges.")
        return edges

    def detect_from_graph(self, graph) -> List[ConflictEdge]:
        """
        Convenience wrapper: extract triples from a NetworkX DiGraph
        where edges have 'predicate', 'object', and 'source_doc' attributes.

        Expected graph format:
            G.nodes[n] = {"label": "entity_name"}
            G.edges[u, v] = {"predicate": "...", "source_doc": "..."}
        Here u = subject node, v = object node.
        """
        triples = []
        for u, v, data in graph.edges(data=True):
            subject = graph.nodes[u].get("label", str(u))
            obj = graph.nodes[v].get("label", str(v))
            predicate = data.get("predicate", "related_to")
            source = data.get("source_doc", "")
            triples.append(Triple(subject=subject, predicate=predicate, obj=obj, source_doc=source))
        return self.detect(triples)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _shared_predicate(pair: CandidatePair) -> str:
        """
        Return the predicate label for the conflict edge.
        Prefers an exact match; otherwise returns triple_a's predicate.
        """
        pa = pair.triple_a.predicate.strip().lower()
        pb = pair.triple_b.predicate.strip().lower()
        return pair.triple_a.predicate if pa == pb else f"{pair.triple_a.predicate} / {pair.triple_b.predicate}"
