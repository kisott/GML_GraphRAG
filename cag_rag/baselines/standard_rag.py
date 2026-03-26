"""
Standard RAG Baseline
----------------------
Retrieves the top-k most relevant text passages using dense embedding
similarity (no graph structure), then generates an answer with the LLM.

This is the simplest baseline: flat retrieval over raw document chunks.
Conflict handling: none — the LLM receives the raw passages and must
resolve any contradictions implicitly (or fail to notice them).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from ..graph.builder import Document
from ..generation.generator import GenerationResult, OpenAIGenerator, _detect_conflict_flag

logger = logging.getLogger(__name__)


@dataclass
class StandardRAGResult:
    """Output of one Standard RAG retrieval + generation call."""
    query: str
    retrieved_passages: List[str]
    answer: str
    flagged_conflict: bool
    prompt_tokens: int = 0
    completion_tokens: int = 0


class StandardRAG:
    """
    Standard RAG: embed documents, retrieve top-k by cosine similarity,
    generate answer with OpenAI.

    Parameters
    ----------
    documents : List[Document]
        The document corpus.
    generator : OpenAIGenerator
        Shared LLM backend.
    top_k : int
        Number of passages to retrieve.
    embedding_model : str
        SentenceTransformer model for encoding.
    """

    _SYSTEM_PROMPT = (
        "You are a factual question-answering assistant. "
        "Answer the question using ONLY the provided passages. "
        "If the passages contain conflicting information, pick the most plausible answer. "
        "Be concise."
    )

    def __init__(
        self,
        documents: List[Document],
        generator: OpenAIGenerator,
        top_k: int = 3,
        embedding_model: str = "all-MiniLM-L6-v2",
    ):
        self.generator = generator
        self.top_k = top_k
        self._docs = documents
        self._embeddings: Optional[np.ndarray] = None
        self._model = None
        self._embedding_model_name = embedding_model
        self._build_index()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def retrieve_and_generate(self, query: str) -> StandardRAGResult:
        """Retrieve top-k passages and generate an answer."""
        passages, scores = self._retrieve(query)
        prompt = self._build_prompt(query, passages)
        raw = self.generator._call_with_retry(self._SYSTEM_PROMPT, prompt)
        answer = raw.choices[0].message.content.strip()
        usage = raw.usage
        return StandardRAGResult(
            query=query,
            retrieved_passages=passages,
            answer=answer,
            flagged_conflict=_detect_conflict_flag(answer),
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
        )

    # ------------------------------------------------------------------
    # Index + retrieval
    # ------------------------------------------------------------------

    def _build_index(self):
        from sentence_transformers import SentenceTransformer
        logger.info(f"StandardRAG: encoding {len(self._docs)} documents …")
        self._model = SentenceTransformer(self._embedding_model_name)
        texts = [d.text for d in self._docs]
        self._embeddings = self._model.encode(
            texts, normalize_embeddings=True, show_progress_bar=False
        )

    def _retrieve(self, query: str):
        q_emb = self._model.encode([query], normalize_embeddings=True)[0]
        sims = self._embeddings @ q_emb
        top_idx = np.argsort(sims)[::-1][: self.top_k]
        passages = [self._docs[i].text for i in top_idx]
        scores = [float(sims[i]) for i in top_idx]
        return passages, scores

    @staticmethod
    def _build_prompt(query: str, passages: List[str]) -> str:
        lines = ["## Retrieved Passages"]
        for i, p in enumerate(passages, 1):
            lines.append(f"[{i}] {p}")
        lines.append(f"\n## Question\n{query}\n\n## Answer")
        return "\n".join(lines)
