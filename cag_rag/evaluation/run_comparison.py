"""
Baseline Comparison Runner  —  SQuAD
--------------------------------------
Runs Standard RAG, Standard GraphRAG, and CAG-RAG on a SQuAD subset
with injected contradictions and prints a side-by-side comparison.

Metrics reported per system
---------------------------
  conflict_coverage  : % of answers that explicitly mention a conflict
  avg_prompt_tokens  : average prompt tokens consumed per query
  avg_total_tokens   : average total tokens consumed per query

Run
---
    python -m cag_rag.evaluation.run_comparison

Environment variables required
-------------------------------
    OPENAI_API_KEY
    HF_ENDPOINT=https://hf-mirror.com   (if HuggingFace is blocked)
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import List, Dict

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result record for one query across all systems
# ---------------------------------------------------------------------------

@dataclass
class QueryResult:
    article_title: str
    question: str
    gold_answer: str
    standard_rag_answer: str
    standard_rag_flagged: bool
    standard_graphrag_answer: str
    standard_graphrag_flagged: bool
    cag_rag_answer: str
    cag_rag_flagged: bool
    standard_rag_tokens: int = 0
    standard_graphrag_tokens: int = 0
    cag_rag_tokens: int = 0


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run(
    max_articles: int = 5,
    max_qa_per_article: int = 3,
    injection_rate: float = 0.4,
    top_k: int = 3,
    openai_api_key: str = "",
) -> List[QueryResult]:
    from ..data.squad_loader import SQuADLoader
    from ..graph.builder import KnowledgeGraphBuilder
    from ..conflict.detector import ConflictDetector
    from ..graph.index import AugmentedGraphIndex
    from ..retrieval.conflict_aware import ConflictAwareRetriever
    from ..injection.inject_conflicts import ConflictInjector
    from ..generation.generator import OpenAIGenerator
    from ..baselines.standard_rag import StandardRAG
    from ..baselines.standard_graphrag import StandardGraphRAG

    api_key = openai_api_key or os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        raise ValueError("Set OPENAI_API_KEY environment variable.")

    # ------------------------------------------------------------------
    # Shared components
    # ------------------------------------------------------------------
    generator   = OpenAIGenerator(api_key=api_key)
    injector    = ConflictInjector(injection_rate=injection_rate, seed=42)
    detector    = ConflictDetector(
        nli_model_name="cross-encoder/nli-deberta-v3-base",
        contradiction_threshold=0.35,
    )

    # ------------------------------------------------------------------
    # Load SQuAD
    # ------------------------------------------------------------------
    print("\n" + "=" * 65)
    print("Loading SQuAD …")
    loader = SQuADLoader(max_articles=max_articles, max_qa_per_article=max_qa_per_article)
    articles = loader.load()
    print(f"  → {len(articles)} articles, "
          f"{sum(len(a.qa_pairs) for a in articles)} QA pairs total")

    all_results: List[QueryResult] = []

    for article in articles:
        print(f"\n{'─'*55}")
        print(f"Article: {article.title}")

        # ------------------------------------------------------------------
        # 1. Extract triples from article paragraphs (spaCy)
        # ------------------------------------------------------------------
        print("  [1/5] Extracting triples (spaCy) …")
        builder = KnowledgeGraphBuilder(backend="spacy")
        _, clean_triples = builder.build(article.documents)
        print(f"         {len(clean_triples)} triples extracted")

        if not clean_triples:
            logger.warning(f"  No triples extracted for '{article.title}', skipping.")
            continue

        # ------------------------------------------------------------------
        # 2. Inject conflicts
        # ------------------------------------------------------------------
        print("  [2/5] Injecting conflicts …")
        injection_result = injector.inject(clean_triples)
        augmented_triples = injection_result.augmented_triples
        print(f"         {injection_result.n_injected} conflicts injected "
              f"(rate={injection_rate:.0%})")

        # ------------------------------------------------------------------
        # 3. Build systems
        # ------------------------------------------------------------------
        print("  [3/5] Building retrieval systems …")

        # Standard RAG — flat embedding over raw paragraphs
        std_rag = StandardRAG(
            documents=article.documents,
            generator=generator,
            top_k=top_k,
        )

        # Standard GraphRAG — graph from clean triples, no conflict edges
        std_graphrag = StandardGraphRAG(
            triples=clean_triples,
            generator=generator,
            top_k_seeds=top_k,
        )

        # CAG-RAG — graph from augmented triples + conflict edges
        print("  [4/5] Running conflict detection for CAG-RAG …")
        conflict_edges = detector.detect(augmented_triples)
        print(f"         {len(conflict_edges)} conflict edge(s) detected")

        base_graph = KnowledgeGraphBuilder._triples_to_graph(augmented_triples)
        index      = AugmentedGraphIndex.from_base_graph(base_graph, conflict_edges)
        cag_retriever = ConflictAwareRetriever(
            index, top_k_seeds=top_k, bfs_depth=2, min_conflict_confidence=0.3
        )

        # ------------------------------------------------------------------
        # 5. Run all systems on each QA pair
        # ------------------------------------------------------------------
        print("  [5/5] Running queries …")
        for qa in article.qa_pairs:
            q = qa.question
            print(f"\n    Q: {q}")

            # Standard RAG
            r_std = std_rag.retrieve_and_generate(q)

            # Standard GraphRAG
            r_graph = std_graphrag.retrieve_and_generate(q)

            # CAG-RAG
            retrieval = cag_retriever.retrieve(q)
            from ..generation.generator import GenerationResult
            r_cag: GenerationResult = generator.generate(q, retrieval, mode="conflict_aware")

            result = QueryResult(
                article_title=article.title,
                question=q,
                gold_answer=qa.gold_answer,
                standard_rag_answer=r_std.answer,
                standard_rag_flagged=r_std.flagged_conflict,
                standard_graphrag_answer=r_graph.answer,
                standard_graphrag_flagged=r_graph.flagged_conflict,
                cag_rag_answer=r_cag.answer,
                cag_rag_flagged=r_cag.flagged_conflict,
                standard_rag_tokens=r_std.prompt_tokens + r_std.completion_tokens,
                standard_graphrag_tokens=r_graph.prompt_tokens + r_graph.completion_tokens,
                cag_rag_tokens=r_cag.prompt_tokens + r_cag.completion_tokens,
            )
            all_results.append(result)

            print(f"      Std-RAG flagged={r_std.flagged_conflict}  "
                  f"Graph-RAG flagged={r_graph.flagged_conflict}  "
                  f"CAG-RAG flagged={r_cag.flagged_conflict}")

    return all_results


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def print_summary(results: List[QueryResult]):
    if not results:
        print("No results.")
        return

    n = len(results)
    systems = {
        "Standard RAG":      ("standard_rag_flagged",      "standard_rag_tokens"),
        "Standard GraphRAG": ("standard_graphrag_flagged", "standard_graphrag_tokens"),
        "CAG-RAG":           ("cag_rag_flagged",           "cag_rag_tokens"),
    }

    print("\n" + "=" * 65)
    print(f"COMPARISON SUMMARY  ({n} queries)")
    print("=" * 65)
    print(f"{'System':<22} {'Conflict Coverage':>18} {'Avg Tokens':>12}")
    print("-" * 55)

    for name, (flag_attr, tok_attr) in systems.items():
        flagged = sum(getattr(r, flag_attr) for r in results)
        coverage = flagged / n
        avg_tok = sum(getattr(r, tok_attr) for r in results) / n
        print(f"{name:<22} {coverage:>17.1%} {avg_tok:>12.0f}")

    print("=" * 65)

    # Per-query detail
    print("\nPER-QUERY DETAIL")
    print(f"{'Question':<40} {'Gold':<15} {'Std-RAG':^9} {'Graph':^9} {'CAG':^9}")
    print("-" * 85)
    for r in results:
        q_short = r.question[:38] + ".." if len(r.question) > 40 else r.question
        g_short = r.gold_answer[:13] + ".." if len(r.gold_answer) > 15 else r.gold_answer
        sr = "FLAG" if r.standard_rag_flagged else "    "
        sg = "FLAG" if r.standard_graphrag_flagged else "    "
        sc = "FLAG" if r.cag_rag_flagged else "    "
        print(f"{q_short:<40} {g_short:<15} {sr:^9} {sg:^9} {sc:^9}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    results = run(
        max_articles=3,
        max_qa_per_article=3,
        injection_rate=0.4,
    )
    print_summary(results)
