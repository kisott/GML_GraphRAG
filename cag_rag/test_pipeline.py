"""
End-to-end smoke test: builder → conflict detector → index → retriever
Run:  python -m cag_rag.test_pipeline
"""

import logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

import networkx as nx

from cag_rag.conflict.semantic_filter import Triple
from cag_rag.conflict.detector import ConflictDetector
from cag_rag.graph.builder import KnowledgeGraphBuilder
from cag_rag.graph.index import AugmentedGraphIndex
from cag_rag.retrieval.conflict_aware import ConflictAwareRetriever

# ---------------------------------------------------------------------------
# 1. Define a small set of triples (simulating extracted knowledge)
# ---------------------------------------------------------------------------
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
    Triple("ETH Zurich",      "located_in",  "Zurich",  source_doc="eth_web", triple_id="t10"),
    Triple("Princeton",       "located_in",  "New Jersey", source_doc="pton", triple_id="t11"),
]


def main():
    print("=" * 65)
    print("CAG-RAG  End-to-End Pipeline  —  Smoke Test")
    print("=" * 65)

    # ------------------------------------------------------------------
    # 2. Build base knowledge graph from triples
    # ------------------------------------------------------------------
    print("\n[1/4] Building base knowledge graph …")
    base_graph = KnowledgeGraphBuilder._triples_to_graph(TRIPLES)
    print(f"  → {base_graph.number_of_nodes()} nodes, {base_graph.number_of_edges()} edges")

    # ------------------------------------------------------------------
    # 3. Detect conflicts
    # ------------------------------------------------------------------
    print("\n[2/4] Running conflict detection …")
    detector = ConflictDetector(
        nli_model_name="cross-encoder/nli-deberta-v3-base",
        contradiction_threshold=0.35,
        use_soft_match=True,
    )
    conflict_edges = detector.detect(TRIPLES)
    print(f"  → {len(conflict_edges)} conflict edge(s) detected")
    for ce in conflict_edges:
        print(f"     {ce}")

    # ------------------------------------------------------------------
    # 4. Build augmented index G' = (V, E ∪ C)
    # ------------------------------------------------------------------
    print("\n[3/4] Building augmented graph index G' …")
    index = AugmentedGraphIndex.from_base_graph(base_graph, conflict_edges)
    print(index.summary())

    # ------------------------------------------------------------------
    # 5. Conflict-aware retrieval
    # ------------------------------------------------------------------
    print("\n[4/4] Running conflict-aware retrieval …")
    retriever = ConflictAwareRetriever(
        index,
        top_k_seeds=3,
        bfs_depth=2,
        min_conflict_confidence=0.0,   # include all detected conflicts in test
    )

    query = "Where was Albert Einstein born?"
    print(f"\n  Query: '{query}'")
    result = retriever.retrieve(query)

    print("\n" + result.to_context_string())

    if result.has_conflicts:
        print(f"\n  ✓ Conflict-aware retrieval surfaced {len(result.conflict_contexts)} conflict(s).")
    else:
        print("\n  (No conflicts surfaced for this query.)")


if __name__ == "__main__":
    main()
