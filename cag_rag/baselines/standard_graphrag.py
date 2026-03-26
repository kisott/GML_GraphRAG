"""
Standard GraphRAG Baseline
---------------------------
Builds a knowledge graph and retrieves relevant subgraphs via BFS,
but WITHOUT conflict edge injection or conflict-aware expansion.

This isolates the effect of graph structure (vs. Standard RAG) and
conflict encoding (vs. CAG-RAG).

Implementation reuses the existing CAG-RAG components with an empty
conflict edge list, so the only difference from CAG-RAG is:
  1. No conflict edges in the index.
  2. LLM receives standard prompt (no conflict blocks).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List

from ..conflict.semantic_filter import Triple
from ..graph.index import AugmentedGraphIndex
from ..retrieval.conflict_aware import ConflictAwareRetriever, RetrievalResult
from ..generation.generator import GenerationResult, OpenAIGenerator

logger = logging.getLogger(__name__)


@dataclass
class StandardGraphRAGResult:
    """Output of one Standard GraphRAG call."""
    query: str
    subgraph_triples: List[dict]
    answer: str
    flagged_conflict: bool
    prompt_tokens: int = 0
    completion_tokens: int = 0


class StandardGraphRAG:
    """
    Standard GraphRAG: graph-based retrieval without conflict awareness.

    Builds G = (V, E) with no conflict edges, retrieves relevant subgraph
    by BFS, and generates an answer using the standard (non-conflict-aware)
    prompt mode.

    Parameters
    ----------
    triples : List[Triple]
        Extracted triples from the corpus (no injected conflicts).
    generator : OpenAIGenerator
        Shared LLM backend.
    top_k_seeds : int
        Seed nodes for BFS retrieval.
    bfs_depth : int
        BFS expansion depth.
    embedding_model : str
        SentenceTransformer model for query→node similarity.
    """

    def __init__(
        self,
        triples: List[Triple],
        generator: OpenAIGenerator,
        top_k_seeds: int = 5,
        bfs_depth: int = 2,
        embedding_model: str = "all-MiniLM-L6-v2",
    ):
        self.generator = generator
        logger.info("StandardGraphRAG: building index (no conflict edges) …")

        # Build graph with empty conflict list — pure relation graph
        from ..graph.builder import KnowledgeGraphBuilder
        base_graph = KnowledgeGraphBuilder._triples_to_graph(triples)
        self._index = AugmentedGraphIndex.from_base_graph(base_graph, conflict_edges=[])
        self._retriever = ConflictAwareRetriever(
            self._index,
            top_k_seeds=top_k_seeds,
            bfs_depth=bfs_depth,
            min_conflict_confidence=1.1,   # effectively disables conflict expansion
            embedding_model=embedding_model,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def retrieve_and_generate(self, query: str) -> StandardGraphRAGResult:
        """Retrieve subgraph and generate an answer (standard mode)."""
        retrieval_result: RetrievalResult = self._retriever.retrieve(query)

        # Generate with standard mode — no conflict context passed to LLM
        gen_result: GenerationResult = self.generator.generate(
            query, retrieval_result, mode="standard"
        )

        return StandardGraphRAGResult(
            query=query,
            subgraph_triples=retrieval_result.subgraph_triples,
            answer=gen_result.answer,
            flagged_conflict=gen_result.flagged_conflict,
            prompt_tokens=gen_result.prompt_tokens,
            completion_tokens=gen_result.completion_tokens,
        )
