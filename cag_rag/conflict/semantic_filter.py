"""
Stage 1: Semantic Similarity Filter
-----------------------------------
Given a list of knowledge-graph triples, find candidate conflicting pairs
by checking that:
  1. They share the same (or semantically similar) subject entity.
  2. They share the same (or semantically similar) predicate.
  3. Their object embeddings are NOT too similar — ruling out paraphrases.

Returns a list of candidate pairs for Stage 2 (NLI classification).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class Triple:
    """A single knowledge-graph triple with provenance."""
    subject: str
    predicate: str
    obj: str                        # 'object' is a Python builtin — avoid shadowing
    source_doc: str = ""            # document / chunk id where this triple was extracted
    triple_id: Optional[str] = None # optional unique identifier

    def as_text(self) -> str:
        """Natural-language rendering used for embedding."""
        return f"{self.subject} {self.predicate} {self.obj}"


@dataclass
class CandidatePair:
    """Two triples that passed the semantic filter and are sent to NLI."""
    triple_a: Triple
    triple_b: Triple
    subject_sim: float              # cosine similarity between subject embeddings
    predicate_sim: float            # cosine similarity between predicate embeddings
    object_sim: float               # cosine similarity between object embeddings
    # populated by NLI stage
    conflict_label: Optional[str] = None   # "contradiction" | "neutral" | "entailment"
    conflict_score: float = 0.0            # NLI contradiction probability


# ---------------------------------------------------------------------------
# Embedding helper (lazy-loaded to avoid import cost when not needed)
# ---------------------------------------------------------------------------

class EmbeddingModel:
    """
    Thin wrapper around a SentenceTransformer model.
    Loaded once and reused for all encode() calls.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self._model_name = model_name
        self._model = None

    def _load(self):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise ImportError(
                "sentence-transformers is required. "
                "Install with: pip install sentence-transformers"
            ) from e
        self._model = SentenceTransformer(self._model_name)

    def encode(self, texts: List[str]) -> np.ndarray:
        if self._model is None:
            self._load()
        return self._model.encode(texts, normalize_embeddings=True, show_progress_bar=False)


# ---------------------------------------------------------------------------
# Semantic Filter
# ---------------------------------------------------------------------------

class SemanticFilter:
    """
    Identifies candidate conflicting triple pairs via two criteria:

    Hard-match mode (fast, exact string matching):
        Triples with identical (subject, predicate) but different objects
        are flagged immediately — no embedding needed.

    Soft-match mode (embedding similarity):
        Triples whose subjects AND predicates are semantically close
        (above `subject_threshold` / `predicate_threshold`) but whose
        objects are NOT too similar (below `object_sim_max`) are
        considered candidate conflicts.

    The two modes are run together; duplicates are deduplicated.
    """

    def __init__(
        self,
        subject_threshold: float = 0.85,
        predicate_threshold: float = 0.80,
        object_sim_max: float = 0.90,   # object pairs above this are paraphrases, not conflicts
        embedding_model_name: str = "all-MiniLM-L6-v2",
        use_soft_match: bool = True,
    ):
        self.subject_threshold = subject_threshold
        self.predicate_threshold = predicate_threshold
        self.object_sim_max = object_sim_max
        self.use_soft_match = use_soft_match
        self._embedder = EmbeddingModel(embedding_model_name)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def filter(self, triples: List[Triple]) -> List[CandidatePair]:
        """
        Return candidate conflicting pairs from a list of triples.
        Combines hard-match and (optionally) soft-match strategies.
        """
        candidates: dict[Tuple[int, int], CandidatePair] = {}

        # --- Hard match (no embeddings needed) ---
        for i, j in combinations(range(len(triples)), 2):
            ta, tb = triples[i], triples[j]
            if self._hard_match(ta, tb):
                pair = CandidatePair(
                    triple_a=ta, triple_b=tb,
                    subject_sim=1.0, predicate_sim=1.0, object_sim=0.0,
                )
                candidates[(i, j)] = pair

        # --- Soft match (embedding-based) ---
        if self.use_soft_match and len(triples) > 1:
            soft_pairs = self._soft_match(triples)
            for (i, j), pair in soft_pairs.items():
                if (i, j) not in candidates:
                    candidates[(i, j)] = pair

        return list(candidates.values())

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _hard_match(self, ta: Triple, tb: Triple) -> bool:
        """
        Same subject + same predicate (case-insensitive) but different objects.
        Objects must not be exact duplicates.
        """
        same_subject = ta.subject.strip().lower() == tb.subject.strip().lower()
        same_predicate = ta.predicate.strip().lower() == tb.predicate.strip().lower()
        diff_object = ta.obj.strip().lower() != tb.obj.strip().lower()
        return same_subject and same_predicate and diff_object

    def _soft_match(self, triples: List[Triple]) -> dict[Tuple[int, int], CandidatePair]:
        """Embedding-based candidate detection."""
        subjects = [t.subject for t in triples]
        predicates = [t.predicate for t in triples]
        objects = [t.obj for t in triples]

        subj_emb = self._embedder.encode(subjects)    # (N, D)
        pred_emb = self._embedder.encode(predicates)  # (N, D)
        obj_emb = self._embedder.encode(objects)      # (N, D)

        # Cosine similarity matrices (embeddings are already L2-normalised)
        subj_sim = subj_emb @ subj_emb.T   # (N, N)
        pred_sim = pred_emb @ pred_emb.T
        obj_sim = obj_emb @ obj_emb.T

        candidates: dict[Tuple[int, int], CandidatePair] = {}
        for i, j in combinations(range(len(triples)), 2):
            s_sim = float(subj_sim[i, j])
            p_sim = float(pred_sim[i, j])
            o_sim = float(obj_sim[i, j])

            if (
                s_sim >= self.subject_threshold
                and p_sim >= self.predicate_threshold
                and o_sim < self.object_sim_max
            ):
                candidates[(i, j)] = CandidatePair(
                    triple_a=triples[i], triple_b=triples[j],
                    subject_sim=s_sim, predicate_sim=p_sim, object_sim=o_sim,
                )
        return candidates
