"""
End-to-end test: injection → detection → retrieval → generation
Run:  python -m cag_rag.generation.test_generation
Requires GEMINI_API_KEY environment variable.
"""

import logging
import os
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

from cag_rag.conflict.semantic_filter import Triple
from cag_rag.conflict.detector import ConflictDetector
from cag_rag.graph.builder import KnowledgeGraphBuilder
from cag_rag.graph.index import AugmentedGraphIndex
from cag_rag.retrieval.conflict_aware import ConflictAwareRetriever
from cag_rag.generation.generator import OpenAIGenerator

TRIPLES = [
    Triple("Albert Einstein", "born_in",     "Ulm",     source_doc="wiki_v1", triple_id="t1"),
    Triple("Albert Einstein", "born_in",     "Munich",  source_doc="wiki_v2", triple_id="t2"),
    Triple("Albert Einstein", "nationality", "German",  source_doc="wiki_v1", triple_id="t3"),
    Triple("Albert Einstein", "worked_at",   "ETH Zurich", source_doc="doc_a", triple_id="t4"),
    Triple("Albert Einstein", "worked_at",   "Princeton",  source_doc="doc_b", triple_id="t5"),
    Triple("Marie Curie",     "born_in",     "Warsaw",  source_doc="bio_c",  triple_id="t6"),
    Triple("Marie Curie",     "nationality", "Polish",  source_doc="bio_c",  triple_id="t7"),
    Triple("Marie Curie",     "award",       "Nobel Prize in Physics",   source_doc="doc_x", triple_id="t8"),
    Triple("Marie Curie",     "award",       "Nobel Prize in Chemistry", source_doc="doc_y", triple_id="t9"),
]

QUERY = "Where was Albert Einstein born?"


def main():
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        print("ERROR: set OPENAI_API_KEY environment variable first.")
        return

    print("=" * 65)
    print("CAG-RAG  Full Pipeline  —  Generation Test")
    print("=" * 65)

    # 1. Build graph + detect conflicts
    print("\n[1/4] Building graph and detecting conflicts …")
    base_graph = KnowledgeGraphBuilder._triples_to_graph(TRIPLES)
    detector = ConflictDetector(
        nli_model_name="cross-encoder/nli-deberta-v3-base",
        contradiction_threshold=0.35,
    )
    conflict_edges = detector.detect(TRIPLES)
    print(f"  → {len(conflict_edges)} conflict(s) detected")

    # 2. Build augmented index
    print("\n[2/4] Building augmented index …")
    index = AugmentedGraphIndex.from_base_graph(base_graph, conflict_edges)
    print(index.summary())

    # 3. Retrieve
    print(f"\n[3/4] Retrieving context for: '{QUERY}'")
    retriever = ConflictAwareRetriever(index, top_k_seeds=3, bfs_depth=2)
    retrieval_result = retriever.retrieve(QUERY)
    print(retrieval_result.to_context_string())

    # 4. Generate — both modes for comparison
    print("\n[4/4] Generating answers …")
    gen = OpenAIGenerator(api_key=api_key)
    results = gen.generate_both_modes(QUERY, retrieval_result)

    for mode, res in results.items():
        print(f"\n{'─'*55}")
        print(f"Mode: {mode.upper()}")
        print(f"Answer: {res.answer}")
        print(f"Conflict flagged: {res.flagged_conflict}")
        print(f"Tokens: {res.prompt_tokens} prompt + {res.completion_tokens} completion")


if __name__ == "__main__":
    main()
