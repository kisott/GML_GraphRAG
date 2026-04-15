"""
TruthfulRAG Baseline
---------------------
A retrieval-augmented generation system that resolves conflicts at
*retrieval time* using NLI-based consistency filtering, rather than
exposing them to the LLM as CAG-RAG does.

Algorithm (entropy-based consistency filtering)
------------------------------------------------
1. Retrieve a candidate pool of top_k * pool_factor passages by embedding
   similarity (same as Standard RAG, but larger pool).

2. Cross-check every pair of candidate passages with an NLI model
   (cross-encoder/nli-deberta-v3-base) to get pairwise scores:
     consistency(p_i, p_j) = entailment_prob(p_i → p_j)
                            + entailment_prob(p_j → p_i)  (symmetric)

3. Score each passage by its *mean consistency* with all other passages:
     score(p_i) = mean_j(consistency(p_i, p_j))
   High-scoring passages agree with the majority; low-scoring passages
   are outliers (likely to contain conflicting information).

4. Select the top_k passages by consistency score.

5. Generate the answer with the standard (non-conflict-aware) prompt.

Comparison role
---------------
TruthfulRAG represents generation-time conflict suppression:
it silently filters contradictory evidence to produce a cleaner, more
consistent answer.  CAG-RAG instead surfaces conflicts explicitly.
This contrast is the core evaluation in the paper.

Usage
-----
    from cag_rag.baselines.truthful_rag import TruthfulRAG

    trag = TruthfulRAG(documents=docs, generator=gen, top_k=3)
    result = trag.retrieve_and_generate("Who won Super Bowl 50?")
    print(result.answer)
    print(result.consistency_scores)   # per-passage scores
    print(result.n_filtered)           # passages removed by filtering
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from ..graph.builder import Document
from ..generation.generator import GenerationResult, OpenAIGenerator, _detect_conflict_flag

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class TruthfulRAGResult:
    """Output of one TruthfulRAG retrieval + generation call."""
    query: str
    candidate_passages: List[str]       # all retrieved before filtering
    filtered_passages: List[str]        # passages kept after consistency filter
    consistency_scores: List[float]     # score per candidate passage
    n_filtered: int                     # number of passages removed
    answer: str
    flagged_conflict: bool
    prompt_tokens: int = 0
    completion_tokens: int = 0


# ---------------------------------------------------------------------------
# TruthfulRAG
# ---------------------------------------------------------------------------

class TruthfulRAG:
    """
    TruthfulRAG: consistency-filtered dense retrieval baseline.

    Retrieves a larger candidate pool, cross-checks passages with NLI,
    and keeps only the most internally consistent subset before generation.

    Parameters
    ----------
    documents : List[Document]
        The document corpus.
    generator : OpenAIGenerator
        Shared LLM backend.
    top_k : int
        Number of passages to keep after filtering.
    pool_factor : int
        Retrieve top_k * pool_factor candidates before filtering.
        Must be >= 2 to allow cross-checking.
    nli_model : str
        Cross-encoder NLI model for consistency scoring.
    embedding_model : str
        SentenceTransformer model for initial retrieval.
    """

    _SYSTEM_PROMPT = (
        "You are a factual question-answering assistant. "
        "Answer the question using ONLY the provided passages. "
        "Be concise and direct."
    )

    def __init__(
        self,
        documents: List[Document],
        generator: OpenAIGenerator,
        top_k: int = 3,
        pool_factor: int = 2,
        nli_model: str = "cross-encoder/nli-deberta-v3-base",
        embedding_model: str = "all-MiniLM-L6-v2",
    ):
        self.generator = generator
        self.top_k = top_k
        self.pool_size = max(top_k * pool_factor, top_k + 1)
        self._docs = documents
        self._nli_model_name = nli_model
        self._embedding_model_name = embedding_model
        self._nli_model = None
        self._emb_model = None
        self._embeddings: Optional[np.ndarray] = None
        self._build_index()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def retrieve_and_generate(self, query: str) -> TruthfulRAGResult:
        """Retrieve, filter for consistency, and generate an answer."""
        # Step 1: retrieve candidate pool
        candidates, _ = self._retrieve(query, k=self.pool_size)

        # Step 2: NLI consistency scoring
        scores = self._consistency_scores(candidates)

        # Step 3: select top_k by consistency score
        ranked = sorted(zip(scores, candidates), reverse=True)
        kept_pairs = ranked[: self.top_k]
        kept_scores   = [s for s, _ in kept_pairs]
        kept_passages = [p for _, p in kept_pairs]
        n_filtered = len(candidates) - len(kept_passages)

        logger.info(
            f"TruthfulRAG: {len(candidates)} candidates → "
            f"{len(kept_passages)} kept, {n_filtered} filtered"
        )

        # Step 4: generate from filtered passages
        prompt = self._build_prompt(query, kept_passages)
        raw = self.generator._call_with_retry(self._SYSTEM_PROMPT, prompt)
        answer = raw.choices[0].message.content.strip()
        usage = raw.usage

        # Attach per-candidate scores in original retrieval order
        all_scores = [s for s, _ in ranked]

        return TruthfulRAGResult(
            query=query,
            candidate_passages=candidates,
            filtered_passages=kept_passages,
            consistency_scores=all_scores,
            n_filtered=n_filtered,
            answer=answer,
            flagged_conflict=_detect_conflict_flag(answer),
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
        )

    # ------------------------------------------------------------------
    # Embedding index
    # ------------------------------------------------------------------

    def _build_index(self):
        from sentence_transformers import SentenceTransformer
        logger.info(f"TruthfulRAG: encoding {len(self._docs)} documents …")
        self._emb_model = SentenceTransformer(self._embedding_model_name)
        texts = [d.text for d in self._docs]
        self._embeddings = self._emb_model.encode(
            texts, normalize_embeddings=True, show_progress_bar=False
        )

    def _retrieve(self, query: str, k: int) -> Tuple[List[str], List[float]]:
        q_emb = self._emb_model.encode([query], normalize_embeddings=True)[0]
        sims = self._embeddings @ q_emb
        top_idx = np.argsort(sims)[::-1][:k]
        passages = [self._docs[i].text for i in top_idx]
        scores   = [float(sims[i]) for i in top_idx]
        return passages, scores

    # ------------------------------------------------------------------
    # NLI-based consistency scoring
    # ------------------------------------------------------------------

    def _load_nli(self):
        from sentence_transformers import CrossEncoder
        logger.info(f"TruthfulRAG: loading NLI model {self._nli_model_name} …")
        self._nli_model = CrossEncoder(self._nli_model_name)

    def _consistency_scores(self, passages: List[str]) -> List[float]:
        """
        For each passage, compute its mean pairwise consistency with all others.
        consistency(p_i, p_j) = entailment_i→j + entailment_j→i  (0–2 range)
        score(p_i) = mean over j≠i of consistency(p_i, p_j)

        If only one passage, returns [1.0].
        """
        n = len(passages)
        if n <= 1:
            return [1.0] * n

        if self._nli_model is None:
            self._load_nli()

        # Build all ordered pairs (i→j) and (j→i)
        pairs: List[Tuple[str, str]] = []
        pair_indices: List[Tuple[int, int]] = []
        for i in range(n):
            for j in range(n):
                if i != j:
                    pairs.append((passages[i], passages[j]))
                    pair_indices.append((i, j))

        # Batch NLI inference
        import numpy as np
        from scipy.special import softmax

        logits = self._nli_model.predict(pairs)
        if logits.ndim == 1:
            logits = logits[np.newaxis, :]
        probs = softmax(logits, axis=1)

        # Label order: contradiction=0, entailment=1, neutral=2
        # (cross-encoder/nli-deberta-v3-base)
        entailment_idx = 1

        # Accumulate entailment probability for each passage
        entail_sum = np.zeros(n)
        count = np.zeros(n)
        for k_idx, (i, j) in enumerate(pair_indices):
            entail_prob = float(probs[k_idx, entailment_idx])
            entail_sum[i] += entail_prob
            count[i] += 1

        scores = (entail_sum / np.maximum(count, 1)).tolist()
        logger.info(f"TruthfulRAG consistency scores: {[f'{s:.3f}' for s in scores]}")
        return scores

    # ------------------------------------------------------------------
    # Prompt builder
    # ------------------------------------------------------------------

    @staticmethod
    def _build_prompt(query: str, passages: List[str]) -> str:
        lines = ["## Retrieved Passages (consistency-filtered)"]
        for i, p in enumerate(passages, 1):
            lines.append(f"[{i}] {p}")
        lines.append(f"\n## Question\n{query}\n\n## Answer")
        return "\n".join(lines)
