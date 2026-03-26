"""
Quick smoke-test for the conflict detection module.
Run with:  python -m cag_rag.conflict.test_detector
"""

import logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

from cag_rag.conflict import ConflictDetector, Triple

# ---------------------------------------------------------------------------
# Sample triples — mix of real conflicts and non-conflicts
# ---------------------------------------------------------------------------
TRIPLES = [
    # Conflict: same (subject, predicate), different objects
    Triple("Albert Einstein", "born_in",    "Ulm",       source_doc="wikipedia_v1", triple_id="t1"),
    Triple("Albert Einstein", "born_in",    "Munich",    source_doc="wiki_mirror",  triple_id="t2"),

    # Conflict: temporal
    Triple("World War II",    "ended_in",   "1945",      source_doc="history_a",    triple_id="t3"),
    Triple("World War II",    "ended_in",   "1944",      source_doc="history_b",    triple_id="t4"),

    # Non-conflict: same (subject, predicate, object)
    Triple("Marie Curie",     "nationality","Polish",    source_doc="doc_x",        triple_id="t5"),
    Triple("Marie Curie",     "nationality","Polish",    source_doc="doc_y",        triple_id="t6"),

    # Non-conflict: different subjects
    Triple("Newton",          "born_in",    "Woolsthorpe", source_doc="doc_n",      triple_id="t7"),
    Triple("Leibniz",         "born_in",    "Leipzig",   source_doc="doc_l",        triple_id="t8"),

    # Conflict: attribute (numerical)
    Triple("Eiffel Tower",    "height_metres", "300",   source_doc="source_a",      triple_id="t9"),
    Triple("Eiffel Tower",    "height_metres", "324",   source_doc="source_b",      triple_id="t10"),
]


def main():
    print("=" * 60)
    print("CAG-RAG  Conflict Detection  —  Smoke Test")
    print("=" * 60)

    # Use a lighter NLI model for quick testing
    detector = ConflictDetector(
        nli_model_name="cross-encoder/nli-deberta-v3-base",   # smaller than large
        contradiction_threshold=0.4,
        use_soft_match=True,
    )

    edges = detector.detect(TRIPLES)

    if not edges:
        print("No conflicts detected.")
    else:
        print(f"\nDetected {len(edges)} conflict edge(s):\n")
        for i, edge in enumerate(edges, 1):
            print(f"  [{i}] {edge}")
            print(f"       edge_id  : {edge.edge_id}")
            print(f"       type     : {edge.conflict_type}")
            print(f"       confidence: {edge.detection_confidence:.3f}")
            print(f"       sources  : '{edge.source_a}'  vs  '{edge.source_b}'")
            print()


if __name__ == "__main__":
    main()
