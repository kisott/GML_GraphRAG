"""
Baseline Comparison Runner  —  SQuAD
--------------------------------------
Runs Standard RAG, Standard GraphRAG, and CAG-RAG on a SQuAD subset
with injected contradictions and prints a side-by-side comparison.

Metrics reported per system
---------------------------
  exact_match        : % of answers that exactly match the gold answer (normalized)
  f1                 : average token-level F1 against gold answer
  conflict_coverage  : % of answers that explicitly mention a conflict
  avg_total_tokens   : average total tokens consumed per query

Experiments
-----------
  Single run    : python -m cag_rag.evaluation.run_comparison
  Sweep         : python -m cag_rag.evaluation.run_comparison --sweep
  More queries  : python -m cag_rag.evaluation.run_comparison --articles 10 --qa 5

Environment variables required
-------------------------------
    OPENAI_API_KEY
    HF_ENDPOINT=https://hf-mirror.com   (if HuggingFace is blocked)
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import logging
import os
import re
import string
from dataclasses import dataclass, field
from typing import List, Dict, Optional

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# SQuAD-standard answer normalization and scoring
# ---------------------------------------------------------------------------

def _normalize_answer(s: str) -> str:
    """Lower, strip punctuation/articles/extra whitespace (SQuAD official)."""
    s = s.lower()
    s = s.translate(str.maketrans("", "", string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def exact_match(prediction: str, gold: str) -> bool:
    return _normalize_answer(prediction) == _normalize_answer(gold)


def token_f1(prediction: str, gold: str) -> float:
    pred_tokens = _normalize_answer(prediction).split()
    gold_tokens = _normalize_answer(gold).split()
    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)
    common = collections.Counter(pred_tokens) & collections.Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


# ---------------------------------------------------------------------------
# Result record for one query across all systems
# ---------------------------------------------------------------------------

@dataclass
class QueryResult:
    article_title: str
    question: str
    gold_answer: str
    injection_rate: float
    standard_rag_answer: str
    standard_rag_flagged: bool
    standard_graphrag_answer: str
    standard_graphrag_flagged: bool
    cag_rag_answer: str
    cag_rag_flagged: bool
    # TruthfulRAG (optional — empty string means not run)
    truthful_rag_answer: str = ""
    truthful_rag_flagged: bool = False
    truthful_rag_tokens: int = 0
    truthful_rag_n_filtered: int = 0
    # Token counts
    standard_rag_tokens: int = 0
    standard_graphrag_tokens: int = 0
    cag_rag_tokens: int = 0
    # Accuracy metrics (computed post-hoc)
    standard_rag_em: bool = False
    standard_rag_f1: float = 0.0
    standard_graphrag_em: bool = False
    standard_graphrag_f1: float = 0.0
    cag_rag_em: bool = False
    cag_rag_f1: float = 0.0
    truthful_rag_em: bool = False
    truthful_rag_f1: float = 0.0
    # Faithfulness scores (LLM-as-judge, optional)
    standard_rag_faithfulness: float = -1.0
    standard_graphrag_faithfulness: float = -1.0
    cag_rag_faithfulness: float = -1.0
    truthful_rag_faithfulness: float = -1.0
    # Context strings (stored for faithfulness judge)
    standard_rag_context: str = field(default="", repr=False)
    standard_graphrag_context: str = field(default="", repr=False)
    cag_rag_context: str = field(default="", repr=False)
    truthful_rag_context: str = field(default="", repr=False)

    def compute_accuracy(self):
        """Fill in EM/F1 fields from answers vs gold."""
        self.standard_rag_em      = exact_match(self.standard_rag_answer,      self.gold_answer)
        self.standard_rag_f1      = token_f1(self.standard_rag_answer,         self.gold_answer)
        self.standard_graphrag_em = exact_match(self.standard_graphrag_answer,  self.gold_answer)
        self.standard_graphrag_f1 = token_f1(self.standard_graphrag_answer,     self.gold_answer)
        self.cag_rag_em           = exact_match(self.cag_rag_answer,            self.gold_answer)
        self.cag_rag_f1           = token_f1(self.cag_rag_answer,               self.gold_answer)
        if self.truthful_rag_answer:
            self.truthful_rag_em  = exact_match(self.truthful_rag_answer,       self.gold_answer)
            self.truthful_rag_f1  = token_f1(self.truthful_rag_answer,          self.gold_answer)


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run(
    max_articles: int = 10,
    max_qa_per_article: int = 5,
    injection_rate: float = 0.4,
    top_k: int = 3,
    openai_api_key: str = "",
    include_truthful: bool = True,
    include_faithfulness: bool = False,
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
    from ..baselines.truthful_rag import TruthfulRAG
    from .faithfulness import FaithfulnessScorer

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
    faithfulness_scorer = FaithfulnessScorer(api_key=api_key) if include_faithfulness else None

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

        # TruthfulRAG — consistency-filtered dense retrieval
        trag = TruthfulRAG(
            documents=article.documents,
            generator=generator,
            top_k=top_k,
        ) if include_truthful else None

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
            std_rag_ctx = " | ".join(r_std.retrieved_passages)

            # Standard GraphRAG
            r_graph = std_graphrag.retrieve_and_generate(q)
            graph_ctx = " | ".join(
                f"{t['subject']} {t['predicate']} {t['object']}"
                for t in r_graph.subgraph_triples
            )

            # CAG-RAG
            retrieval = cag_retriever.retrieve(q)
            from ..generation.generator import GenerationResult
            r_cag: GenerationResult = generator.generate(q, retrieval, mode="conflict_aware")
            cag_ctx = retrieval.to_context_string()

            # TruthfulRAG (optional)
            r_trag = trag.retrieve_and_generate(q) if trag else None
            trag_ctx = " | ".join(r_trag.filtered_passages) if r_trag else ""

            result = QueryResult(
                article_title=article.title,
                question=q,
                gold_answer=qa.gold_answer,
                injection_rate=injection_rate,
                standard_rag_answer=r_std.answer,
                standard_rag_flagged=r_std.flagged_conflict,
                standard_graphrag_answer=r_graph.answer,
                standard_graphrag_flagged=r_graph.flagged_conflict,
                cag_rag_answer=r_cag.answer,
                cag_rag_flagged=r_cag.flagged_conflict,
                truthful_rag_answer=r_trag.answer if r_trag else "",
                truthful_rag_flagged=r_trag.flagged_conflict if r_trag else False,
                truthful_rag_tokens=(r_trag.prompt_tokens + r_trag.completion_tokens) if r_trag else 0,
                truthful_rag_n_filtered=r_trag.n_filtered if r_trag else 0,
                standard_rag_tokens=r_std.prompt_tokens + r_std.completion_tokens,
                standard_graphrag_tokens=r_graph.prompt_tokens + r_graph.completion_tokens,
                cag_rag_tokens=r_cag.prompt_tokens + r_cag.completion_tokens,
                standard_rag_context=std_rag_ctx,
                standard_graphrag_context=graph_ctx,
                cag_rag_context=cag_ctx,
                truthful_rag_context=trag_ctx,
            )
            result.compute_accuracy()

            # Faithfulness scoring (optional — costs extra API calls)
            if faithfulness_scorer:
                has_c = retrieval.has_conflicts
                result.standard_rag_faithfulness = faithfulness_scorer.score(
                    q, r_std.answer, std_rag_ctx, has_c).overall
                result.standard_graphrag_faithfulness = faithfulness_scorer.score(
                    q, r_graph.answer, graph_ctx, has_c).overall
                result.cag_rag_faithfulness = faithfulness_scorer.score(
                    q, r_cag.answer, cag_ctx, has_c).overall
                if r_trag:
                    result.truthful_rag_faithfulness = faithfulness_scorer.score(
                        q, r_trag.answer, trag_ctx, has_c).overall

            all_results.append(result)

            trag_info = (f"  TruthfulRAG flagged={r_trag.flagged_conflict}"
                         f" filtered={r_trag.n_filtered}") if r_trag else ""
            print(f"      Std-RAG flagged={r_std.flagged_conflict}  "
                  f"Graph-RAG flagged={r_graph.flagged_conflict}  "
                  f"CAG-RAG flagged={r_cag.flagged_conflict}"
                  f"{trag_info}")

    return all_results


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def print_summary(results: List[QueryResult], label: str = ""):
    if not results:
        print("No results.")
        return

    n = len(results)
    header = f"COMPARISON SUMMARY  ({n} queries)"
    if label:
        header += f"  [{label}]"

    systems = [
        ("Standard RAG",      "standard_rag_flagged",      "standard_rag_tokens",
         "standard_rag_em",   "standard_rag_f1",      "standard_rag_faithfulness"),
        ("TruthfulRAG",       "truthful_rag_flagged",      "truthful_rag_tokens",
         "truthful_rag_em",   "truthful_rag_f1",      "truthful_rag_faithfulness"),
        ("Standard GraphRAG", "standard_graphrag_flagged", "standard_graphrag_tokens",
         "standard_graphrag_em", "standard_graphrag_f1", "standard_graphrag_faithfulness"),
        ("CAG-RAG",           "cag_rag_flagged",           "cag_rag_tokens",
         "cag_rag_em",        "cag_rag_f1",           "cag_rag_faithfulness"),
    ]

    # Check whether TruthfulRAG and faithfulness were actually run
    has_truthful    = any(r.truthful_rag_answer for r in results)
    has_faithfulness = any(r.cag_rag_faithfulness >= 0 for r in results)

    col_hdr = f"{'System':<22} {'EM':>6} {'F1':>7} {'Conflict Cov':>14} {'Avg Tokens':>12}"
    col_width = 65
    if has_faithfulness:
        col_hdr += f" {'Faithful':>10}"
        col_width += 11

    print("\n" + "=" * col_width)
    print(header)
    print("=" * col_width)
    print(col_hdr)
    print("-" * col_width)

    for name, flag_attr, tok_attr, em_attr, f1_attr, faith_attr in systems:
        if name == "TruthfulRAG" and not has_truthful:
            continue
        em_score  = sum(getattr(r, em_attr)   for r in results) / n
        f1_score  = sum(getattr(r, f1_attr)   for r in results) / n
        flagged   = sum(getattr(r, flag_attr) for r in results)
        coverage  = flagged / n
        avg_tok   = sum(getattr(r, tok_attr)  for r in results) / n
        line = (f"{name:<22} {em_score:>5.1%} {f1_score:>6.1%} "
                f"{coverage:>13.1%} {avg_tok:>12.0f}")
        if has_faithfulness:
            faith_vals = [getattr(r, faith_attr) for r in results if getattr(r, faith_attr) >= 0]
            faith_avg  = sum(faith_vals) / len(faith_vals) if faith_vals else 0.0
            line += f" {faith_avg:>10.2f}"
        print(line)

    print("=" * col_width)

    # Per-query detail
    print("\nPER-QUERY DETAIL")
    print(f"{'Question':<38} {'Gold':<12} "
          f"{'Std-RAG':^11} {'Graph':^11} {'CAG':^11}")
    print("-" * 90)
    for r in results:
        q_short = r.question[:36] + ".." if len(r.question) > 38 else r.question
        g_short = r.gold_answer[:10] + ".." if len(r.gold_answer) > 12 else r.gold_answer

        def _cell(flagged, em, f1):
            parts = []
            if flagged: parts.append("F")
            parts.append(f"f1={f1:.2f}")
            return "/".join(parts)

        sr = _cell(r.standard_rag_flagged,    r.standard_rag_em,    r.standard_rag_f1)
        sg = _cell(r.standard_graphrag_flagged, r.standard_graphrag_em, r.standard_graphrag_f1)
        sc = _cell(r.cag_rag_flagged,         r.cag_rag_em,         r.cag_rag_f1)
        print(f"{q_short:<38} {g_short:<12} {sr:^11} {sg:^11} {sc:^11}")


# ---------------------------------------------------------------------------
# Injection-rate sweep experiment
# ---------------------------------------------------------------------------

def run_injection_sweep(
    rates: List[float] = None,
    max_articles: int = 10,
    max_qa_per_article: int = 5,
    top_k: int = 3,
    openai_api_key: str = "",
    include_truthful: bool = True,
    include_faithfulness: bool = False,
) -> Dict[float, List[QueryResult]]:
    """
    Run the comparison at multiple injection rates and return all results.
    Useful for plotting conflict coverage vs. injection rate.
    """
    if rates is None:
        rates = [0.0, 0.2, 0.4, 0.6]

    sweep_results: Dict[float, List[QueryResult]] = {}
    for rate in rates:
        print(f"\n{'#'*65}")
        print(f"# INJECTION RATE = {rate:.0%}")
        print(f"{'#'*65}")
        sweep_results[rate] = run(
            max_articles=max_articles,
            max_qa_per_article=max_qa_per_article,
            injection_rate=rate,
            top_k=top_k,
            openai_api_key=openai_api_key,
            include_truthful=include_truthful,
            include_faithfulness=include_faithfulness,
        )
        print_summary(sweep_results[rate], label=f"rate={rate:.0%}")

    # Print sweep summary table
    has_truthful = any(
        r.truthful_rag_answer
        for res_list in sweep_results.values()
        for r in res_list
    )
    print("\n" + "=" * 90)
    print("INJECTION-RATE SWEEP SUMMARY")
    print("=" * 90)
    hdr = (f"{'Rate':>6} | {'Std-RAG F1':>10} {'Truthful F1':>11} "
           f"{'Graph F1':>9} {'CAG F1':>8} {'CAG Cov':>9}")
    if not has_truthful:
        hdr = (f"{'Rate':>6} | {'Std-RAG F1':>10} "
               f"{'Graph F1':>9} {'CAG F1':>8} {'CAG Cov':>9}")
    print(hdr)
    print("-" * 90)
    for rate, results in sweep_results.items():
        n = len(results)
        if not n:
            continue
        sr_f1  = sum(r.standard_rag_f1     for r in results) / n
        tr_f1  = sum(r.truthful_rag_f1     for r in results) / n
        sg_f1  = sum(r.standard_graphrag_f1 for r in results) / n
        ca_f1  = sum(r.cag_rag_f1          for r in results) / n
        ca_cov = sum(r.cag_rag_flagged     for r in results) / n
        if has_truthful:
            print(f"{rate:>5.0%} | {sr_f1:>10.1%} {tr_f1:>11.1%} "
                  f"{sg_f1:>9.1%} {ca_f1:>8.1%} {ca_cov:>9.1%}")
        else:
            print(f"{rate:>5.0%} | {sr_f1:>10.1%} "
                  f"{sg_f1:>9.1%} {ca_f1:>8.1%} {ca_cov:>9.1%}")
    print("=" * 90)
    return sweep_results


# ---------------------------------------------------------------------------
# Export helpers
# ---------------------------------------------------------------------------

def save_csv(results: List[QueryResult], path: str):
    """Save per-query results to CSV."""
    if not results:
        return
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fieldnames = list(results[0].__dataclass_fields__.keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow({k: getattr(r, k) for k in fieldnames})
    print(f"Results saved → {path}")


def save_json(results: List[QueryResult], path: str):
    """Save per-query results to JSON."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    data = [{k: getattr(r, k) for k in r.__dataclass_fields__} for r in results]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"Results saved → {path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CAG-RAG baseline comparison")
    parser.add_argument("--articles",       type=int,   default=10,  help="Max SQuAD articles (default 10)")
    parser.add_argument("--qa",             type=int,   default=5,   help="Max QA pairs per article (default 5)")
    parser.add_argument("--rate",           type=float, default=0.4, help="Injection rate for single run (default 0.4)")
    parser.add_argument("--top-k",          type=int,   default=3,   help="Retrieval top-k (default 3)")
    parser.add_argument("--sweep",          action="store_true",     help="Run injection-rate sweep (0%, 20%, 40%, 60%)")
    parser.add_argument("--no-truthful",    action="store_true",     help="Skip TruthfulRAG baseline")
    parser.add_argument("--faithfulness",   action="store_true",     help="Run LLM-as-judge faithfulness scoring (extra API calls)")
    parser.add_argument("--out",            type=str,   default="",  help="Output prefix for CSV/JSON (e.g. results/run1)")
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY", "")

    if args.sweep:
        all_sweep = run_injection_sweep(
            rates=[0.0, 0.2, 0.4, 0.6],
            max_articles=args.articles,
            max_qa_per_article=args.qa,
            top_k=args.top_k,
            openai_api_key=api_key,
            include_truthful=not args.no_truthful,
            include_faithfulness=args.faithfulness,
        )
        if args.out:
            for rate, results in all_sweep.items():
                prefix = f"{args.out}_rate{int(rate*100)}"
                save_csv(results,  prefix + ".csv")
                save_json(results, prefix + ".json")
    else:
        results = run(
            max_articles=args.articles,
            max_qa_per_article=args.qa,
            injection_rate=args.rate,
            top_k=args.top_k,
            openai_api_key=api_key,
            include_truthful=not args.no_truthful,
            include_faithfulness=args.faithfulness,
        )
        print_summary(results)
        if args.out:
            save_csv(results,  args.out + ".csv")
            save_json(results, args.out + ".json")
