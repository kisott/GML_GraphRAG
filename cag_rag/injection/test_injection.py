"""
Smoke test for ConflictInjector.
Run:  python -m cag_rag.injection.test_injection
"""

import logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

from cag_rag.conflict.semantic_filter import Triple
from cag_rag.conflict.detector import ConflictDetector
from cag_rag.injection.inject_conflicts import ConflictInjector, match_detected_to_ground_truth

# ---------------------------------------------------------------------------
# Sample corpus (same as test_pipeline, but no pre-baked conflicts)
# ---------------------------------------------------------------------------
CLEAN_TRIPLES = [
    Triple("Albert Einstein", "born_in",     "Ulm",        source_doc="wiki_v1", triple_id="t1"),
    Triple("Albert Einstein", "nationality", "German",     source_doc="wiki_v1", triple_id="t2"),
    Triple("Albert Einstein", "worked_at",   "ETH Zurich", source_doc="doc_a",  triple_id="t3"),
    Triple("Albert Einstein", "died_in",     "Princeton",  source_doc="doc_a",  triple_id="t4"),
    Triple("Marie Curie",     "born_in",     "Warsaw",     source_doc="bio_c",  triple_id="t5"),
    Triple("Marie Curie",     "nationality", "Polish",     source_doc="bio_c",  triple_id="t6"),
    Triple("Marie Curie",     "award",       "Nobel Prize in Physics",   source_doc="doc_x", triple_id="t7"),
    Triple("Isaac Newton",    "born_in",     "Woolsthorpe", source_doc="hist_a", triple_id="t8"),
    Triple("Isaac Newton",    "nationality", "British",    source_doc="hist_a",  triple_id="t9"),
    Triple("Isaac Newton",    "worked_at",   "Cambridge",  source_doc="hist_a",  triple_id="t10"),
    Triple("ETH Zurich",      "located_in",  "Zurich",     source_doc="eth_web", triple_id="t11"),
    Triple("Cambridge",       "located_in",  "England",    source_doc="cam_web", triple_id="t12"),
    Triple("Princeton",       "located_in",  "New Jersey", source_doc="pton",    triple_id="t13"),
]


def main():
    print("=" * 65)
    print("ConflictInjector  —  Smoke Test")
    print("=" * 65)

    # ------------------------------------------------------------------
    # 1. Inject contradictions
    # ------------------------------------------------------------------
    print("\n[1/3] Injecting contradictions …")
    injector = ConflictInjector(injection_rate=0.4, seed=42)
    result = injector.inject(CLEAN_TRIPLES)
    print(result.summary())

    print("\nGround-truth injected conflicts:")
    for gt in result.ground_truth:
        print(f"  {gt}")

    # ------------------------------------------------------------------
    # 2. Run conflict detector on augmented triples
    # ------------------------------------------------------------------
    print("\n[2/3] Running conflict detector on augmented triples …")
    detector = ConflictDetector(
        nli_model_name="cross-encoder/nli-deberta-v3-base",
        contradiction_threshold=0.35,
        use_soft_match=True,
    )
    detected = detector.detect(result.augmented_triples)
    print(f"  → {len(detected)} conflict edge(s) detected")
    for ce in detected:
        print(f"     {ce}")

    # ------------------------------------------------------------------
    # 3. Evaluate: compare detected vs ground truth
    # ------------------------------------------------------------------
    print("\n[3/3] Evaluation metrics …")
    metrics = match_detected_to_ground_truth(result.ground_truth, detected)
    print(f"  Ground truth conflicts : {metrics['n_ground_truth']}")
    print(f"  Detected conflicts     : {metrics['n_detected']}")
    print(f"  True  Positives (TP)   : {metrics['tp']}")
    print(f"  False Positives (FP)   : {metrics['fp']}")
    print(f"  False Negatives (FN)   : {metrics['fn']}")
    print(f"  Precision              : {metrics['precision']:.4f}")
    print(f"  Recall                 : {metrics['recall']:.4f}")
    print(f"  F1                     : {metrics['f1']:.4f}")


if __name__ == "__main__":
    main()
