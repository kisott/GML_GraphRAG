"""
Result Analysis Charts
-----------------------
Generates publication-ready charts from run_comparison.py CSV output.

Charts produced
---------------
1. plot_injection_sweep  — line chart: injection rate vs. F1 / Conflict Coverage
                           one line per system, two subplots (F1 and Conflict Cov)

2. plot_metric_comparison — grouped bar chart: all metrics for a single run
                            bars grouped by metric, coloured by system

3. plot_faithfulness      — bar chart: faithfulness scores per system
                            (only when --faithfulness results are present)

Usage
-----
    # After running the sweep:
    python -m cag_rag.evaluation.run_comparison --sweep --out results/sweep

    # Then plot:
    python -m cag_rag.visualization.plot_results --sweep-dir results --out figs

    # Or from code:
    from cag_rag.visualization.plot_results import plot_injection_sweep
    plot_injection_sweep("results", output_dir="figs")
"""

from __future__ import annotations

import argparse
import csv
import glob
import logging
import os
import re
from collections import defaultdict
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Colour palette (consistent with graph_viz)
# ---------------------------------------------------------------------------
_SYSTEM_COLORS = {
    "Standard RAG":      "#4C72B0",   # blue
    "TruthfulRAG":       "#55A868",   # green
    "Standard GraphRAG": "#C44E52",   # red
    "CAG-RAG":           "#DD8452",   # orange
}
_SYSTEM_MARKERS = {
    "Standard RAG":      "o",
    "TruthfulRAG":       "s",
    "Standard GraphRAG": "^",
    "CAG-RAG":           "D",
}

_FIELD_MAP = {
    "Standard RAG":      ("standard_rag_f1",      "standard_rag_flagged",      "standard_rag_faithfulness"),
    "TruthfulRAG":       ("truthful_rag_f1",       "truthful_rag_flagged",      "truthful_rag_faithfulness"),
    "Standard GraphRAG": ("standard_graphrag_f1",  "standard_graphrag_flagged", "standard_graphrag_faithfulness"),
    "CAG-RAG":           ("cag_rag_f1",            "cag_rag_flagged",           "cag_rag_faithfulness"),
}


# ---------------------------------------------------------------------------
# CSV loading helpers
# ---------------------------------------------------------------------------

def _load_csv(path: str) -> List[Dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _safe_float(val: str, default: float = 0.0) -> float:
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


def _safe_bool(val: str) -> bool:
    return str(val).strip().lower() in ("true", "1", "yes")


def _aggregate(rows: List[Dict], f1_field: str, flag_field: str, faith_field: str):
    """Return (mean_f1, conflict_coverage, mean_faithfulness) for a list of rows."""
    n = len(rows)
    if not n:
        return 0.0, 0.0, -1.0
    f1   = sum(_safe_float(r.get(f1_field, "0")) for r in rows) / n
    cov  = sum(_safe_bool(r.get(flag_field, "False")) for r in rows) / n
    faith_vals = [_safe_float(r.get(faith_field, "-1")) for r in rows
                  if _safe_float(r.get(faith_field, "-1")) >= 0]
    faith = sum(faith_vals) / len(faith_vals) if faith_vals else -1.0
    return f1, cov, faith


# ---------------------------------------------------------------------------
# 1. plot_injection_sweep
# ---------------------------------------------------------------------------

def plot_injection_sweep(
    sweep_dir: str,
    output_dir: str = "figs",
    filename_prefix: str = "sweep",
) -> List[str]:
    """
    Read sweep CSVs (named *_rate0.csv, *_rate20.csv, …) and plot:
      - Left subplot:  injection rate vs. token F1
      - Right subplot: injection rate vs. conflict coverage

    Parameters
    ----------
    sweep_dir : str
        Directory containing sweep CSV files.
    output_dir : str
        Where to save the PNG.
    filename_prefix : str
        CSV filename prefix (e.g. "sweep" matches sweep_rate0.csv).

    Returns
    -------
    List of saved file paths.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Find and parse sweep CSVs
    pattern = os.path.join(sweep_dir, f"{filename_prefix}_rate*.csv")
    csv_files = sorted(glob.glob(pattern))
    if not csv_files:
        # Try without prefix
        csv_files = sorted(glob.glob(os.path.join(sweep_dir, "*_rate*.csv")))
    if not csv_files:
        logger.warning(f"No sweep CSV files found in {sweep_dir}")
        return []

    # Extract rate from filename (e.g. sweep_rate40.csv → 0.40)
    rate_data: Dict[float, List[Dict]] = {}
    for path in csv_files:
        m = re.search(r"_rate(\d+)", os.path.basename(path))
        rate = int(m.group(1)) / 100.0 if m else 0.0
        rate_data[rate] = _load_csv(path)

    rates = sorted(rate_data.keys())

    # Detect which systems are present
    systems = list(_SYSTEM_COLORS.keys())
    sample_rows = next(iter(rate_data.values()))
    if sample_rows and not sample_rows[0].get("truthful_rag_answer", ""):
        systems = [s for s in systems if s != "TruthfulRAG"]

    # Collect per-rate metrics
    f1_by_system:  Dict[str, List[float]] = defaultdict(list)
    cov_by_system: Dict[str, List[float]] = defaultdict(list)

    for rate in rates:
        rows = rate_data[rate]
        for sys_name in systems:
            f1_field, flag_field, faith_field = _FIELD_MAP[sys_name]
            f1, cov, _ = _aggregate(rows, f1_field, flag_field, faith_field)
            f1_by_system[sys_name].append(f1)
            cov_by_system[sys_name].append(cov)

    rate_pct = [r * 100 for r in rates]

    # --- Plot ---
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for sys_name in systems:
        col = _SYSTEM_COLORS[sys_name]
        mk  = _SYSTEM_MARKERS[sys_name]
        axes[0].plot(rate_pct, [v * 100 for v in f1_by_system[sys_name]],
                     marker=mk, color=col, label=sys_name, linewidth=2, markersize=7)
        axes[1].plot(rate_pct, [v * 100 for v in cov_by_system[sys_name]],
                     marker=mk, color=col, label=sys_name, linewidth=2, markersize=7)

    axes[0].set_title("Token-level F1 vs. Injection Rate", fontsize=13)
    axes[0].set_xlabel("Conflict Injection Rate (%)", fontsize=11)
    axes[0].set_ylabel("Token F1 (%)", fontsize=11)
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xticks(rate_pct)
    axes[0].yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))

    axes[1].set_title("Conflict Coverage vs. Injection Rate", fontsize=13)
    axes[1].set_xlabel("Conflict Injection Rate (%)", fontsize=11)
    axes[1].set_ylabel("Conflict Coverage (%)", fontsize=11)
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xticks(rate_pct)
    axes[1].yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))

    fig.suptitle("CAG-RAG vs. Baselines — Injection Rate Sweep", fontsize=14, y=1.02)
    plt.tight_layout()

    out_path = os.path.join(output_dir, "sweep_line_chart.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved sweep chart → {out_path}")
    return [out_path]


# ---------------------------------------------------------------------------
# 2. plot_metric_comparison
# ---------------------------------------------------------------------------

def plot_metric_comparison(
    csv_path: str,
    output_dir: str = "figs",
) -> List[str]:
    """
    Grouped bar chart from a single run CSV.
    Groups: EM, F1, Conflict Coverage  (+ Faithfulness if present)
    Bars:   one colour per system.

    Parameters
    ----------
    csv_path   : path to a single run_comparison CSV
    output_dir : where to save the PNG

    Returns
    -------
    List of saved file paths.
    """
    os.makedirs(output_dir, exist_ok=True)
    rows = _load_csv(csv_path)
    if not rows:
        logger.warning(f"Empty CSV: {csv_path}")
        return []

    systems = list(_SYSTEM_COLORS.keys())
    if not rows[0].get("truthful_rag_answer", ""):
        systems = [s for s in systems if s != "TruthfulRAG"]

    # Collect metrics per system
    metrics: Dict[str, Dict[str, float]] = {}
    has_faithfulness = False
    for sys_name in systems:
        f1_field, flag_field, faith_field = _FIELD_MAP[sys_name]
        em_field = f1_field.replace("_f1", "_em")
        n = len(rows)
        em  = sum(_safe_bool(r.get(em_field, "False")) for r in rows) / n
        f1  = sum(_safe_float(r.get(f1_field, "0")) for r in rows) / n
        cov = sum(_safe_bool(r.get(flag_field, "False")) for r in rows) / n
        faith_vals = [_safe_float(r.get(faith_field, "-1")) for r in rows
                      if _safe_float(r.get(faith_field, "-1")) >= 0]
        faith = sum(faith_vals) / len(faith_vals) if faith_vals else -1.0
        metrics[sys_name] = {"EM": em, "F1": f1, "Conflict Cov": cov, "Faithfulness": faith}
        if faith >= 0:
            has_faithfulness = True

    metric_names = ["EM", "F1", "Conflict Cov"]
    if has_faithfulness:
        metric_names.append("Faithfulness")

    n_metrics = len(metric_names)
    n_systems = len(systems)
    x = np.arange(n_metrics)
    width = 0.8 / n_systems

    fig, ax = plt.subplots(figsize=(11, 6))

    for i, sys_name in enumerate(systems):
        vals = [metrics[sys_name][m] * 100 for m in metric_names]
        offset = (i - n_systems / 2 + 0.5) * width
        bars = ax.bar(x + offset, vals, width, label=sys_name,
                      color=_SYSTEM_COLORS[sys_name], alpha=0.85, edgecolor="white")
        # Value labels on bars
        for bar, val in zip(bars, vals):
            if val > 1.0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                        f"{val:.1f}", ha="center", va="bottom", fontsize=7.5)

    ax.set_xticks(x)
    ax.set_xticklabels(metric_names, fontsize=12)
    ax.set_ylabel("Score (%)", fontsize=11)
    ax.set_ylim(0, 105)
    ax.set_title("System Comparison — All Metrics", fontsize=13)
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    n_queries = len(rows)
    inj_rate = _safe_float(rows[0].get("injection_rate", "0.4")) * 100
    ax.text(0.01, 0.97, f"n={n_queries} queries  |  injection rate={inj_rate:.0f}%",
            transform=ax.transAxes, fontsize=8, va="top", color="gray")

    plt.tight_layout()
    out_path = os.path.join(output_dir, "metric_comparison_bar.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved metric comparison → {out_path}")
    return [out_path]


# ---------------------------------------------------------------------------
# 3. plot_faithfulness
# ---------------------------------------------------------------------------

def plot_faithfulness(
    csv_path: str,
    output_dir: str = "figs",
) -> List[str]:
    """
    Horizontal bar chart of faithfulness scores per system.
    Shows factual_score and conflict_score as stacked halves.
    Only useful when --faithfulness results are present.
    """
    # Faithfulness detail requires per-system raw scores which aren't in the
    # current CSV (only the overall is stored).  Plot overall faithfulness only.
    os.makedirs(output_dir, exist_ok=True)
    rows = _load_csv(csv_path)
    if not rows:
        return []

    systems = list(_SYSTEM_COLORS.keys())
    if not rows[0].get("truthful_rag_answer", ""):
        systems = [s for s in systems if s != "TruthfulRAG"]

    faith_scores = {}
    for sys_name in systems:
        _, _, faith_field = _FIELD_MAP[sys_name]
        vals = [_safe_float(r.get(faith_field, "-1")) for r in rows
                if _safe_float(r.get(faith_field, "-1")) >= 0]
        if vals:
            faith_scores[sys_name] = sum(vals) / len(vals)

    if not faith_scores:
        logger.warning("No faithfulness scores found in CSV.")
        return []

    names  = list(faith_scores.keys())
    scores = [faith_scores[n] * 100 for n in names]
    colors = [_SYSTEM_COLORS[n] for n in names]

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.barh(names, scores, color=colors, alpha=0.85, edgecolor="white")
    for bar, val in zip(bars, scores):
        ax.text(val + 0.5, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}%", va="center", fontsize=10)

    ax.set_xlim(0, 110)
    ax.set_xlabel("Faithfulness Score (%)", fontsize=11)
    ax.set_title("LLM-as-Judge Faithfulness Score by System", fontsize=13)
    ax.grid(axis="x", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()

    out_path = os.path.join(output_dir, "faithfulness_bar.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved faithfulness chart → {out_path}")
    return [out_path]


# ---------------------------------------------------------------------------
# Demo mode  —  synthetic data based on actual 50-query run results
# ---------------------------------------------------------------------------

def _run_demo(output_dir: str = "figs"):
    """
    Generate all charts using synthetic data that mirrors real experimental
    results (based on the 50-query SQuAD run at injection_rate=0.4).
    Useful for slides before the full evaluation is complete.
    """
    import csv, tempfile, os

    os.makedirs(output_dir, exist_ok=True)

    # --- Sweep data (per-system F1 and conflict coverage at each injection rate) ---
    # Values are realistic estimates consistent with the 50-query run
    sweep_data = {
        0.0: {
            "Standard RAG":      (0.302, 0.00, -1),
            "TruthfulRAG":       (0.278, 0.00, -1),
            "Standard GraphRAG": (0.091, 0.00, -1),
            "CAG-RAG":           (0.075, 0.04, -1),
        },
        0.2: {
            "Standard RAG":      (0.295, 0.00, -1),
            "TruthfulRAG":       (0.270, 0.00, -1),
            "Standard GraphRAG": (0.088, 0.00, -1),
            "CAG-RAG":           (0.068, 0.24, -1),
        },
        0.4: {
            "Standard RAG":      (0.302, 0.00, -1),
            "TruthfulRAG":       (0.261, 0.00, -1),
            "Standard GraphRAG": (0.091, 0.00, -1),
            "CAG-RAG":           (0.059, 0.44, -1),
        },
        0.6: {
            "Standard RAG":      (0.289, 0.00, -1),
            "TruthfulRAG":       (0.248, 0.00, -1),
            "Standard GraphRAG": (0.085, 0.00, -1),
            "CAG-RAG":           (0.051, 0.61, -1),
        },
    }

    # Write synthetic sweep CSVs to a temp dir
    tmpdir = tempfile.mkdtemp()
    fieldnames = [
        "article_title", "question", "gold_answer", "injection_rate",
        "standard_rag_answer", "standard_rag_flagged",
        "standard_graphrag_answer", "standard_graphrag_flagged",
        "cag_rag_answer", "cag_rag_flagged",
        "truthful_rag_answer", "truthful_rag_flagged",
        "standard_rag_tokens", "standard_graphrag_tokens",
        "cag_rag_tokens", "truthful_rag_tokens", "truthful_rag_n_filtered",
        "standard_rag_em", "standard_rag_f1",
        "standard_graphrag_em", "standard_graphrag_f1",
        "cag_rag_em", "cag_rag_f1",
        "truthful_rag_em", "truthful_rag_f1",
        "standard_rag_faithfulness", "standard_graphrag_faithfulness",
        "cag_rag_faithfulness", "truthful_rag_faithfulness",
        "standard_rag_context", "standard_graphrag_context",
        "cag_rag_context", "truthful_rag_context",
    ]

    import random
    rng = random.Random(42)
    n_rows = 50

    for rate, sys_vals in sweep_data.items():
        rows = []
        for i in range(n_rows):
            sr_f1, _, _   = sys_vals["Standard RAG"]
            tr_f1, _, _   = sys_vals["TruthfulRAG"]
            sg_f1, _, _   = sys_vals["Standard GraphRAG"]
            ca_f1, ca_cov, _ = sys_vals["CAG-RAG"]

            def jitter(v, std=0.08): return max(0.0, min(1.0, v + rng.gauss(0, std)))

            row = {
                "article_title": f"Article_{i % 10}",
                "question": f"Question {i}",
                "gold_answer": "answer",
                "injection_rate": rate,
                "standard_rag_answer": "answer text",
                "standard_rag_flagged": "False",
                "standard_graphrag_answer": "answer text",
                "standard_graphrag_flagged": "False",
                "cag_rag_answer": "conflict answer",
                "cag_rag_flagged": str(rng.random() < ca_cov),
                "truthful_rag_answer": "filtered answer",
                "truthful_rag_flagged": "False",
                "standard_rag_tokens": 484,
                "standard_graphrag_tokens": 1395,
                "cag_rag_tokens": 9078,
                "truthful_rag_tokens": 620,
                "truthful_rag_n_filtered": rng.randint(1, 3),
                "standard_rag_em": "False",
                "standard_rag_f1": jitter(sr_f1, 0.12),
                "standard_graphrag_em": "False",
                "standard_graphrag_f1": jitter(sg_f1, 0.06),
                "cag_rag_em": "False",
                "cag_rag_f1": jitter(ca_f1, 0.05),
                "truthful_rag_em": "False",
                "truthful_rag_f1": jitter(tr_f1, 0.11),
                "standard_rag_faithfulness": -1,
                "standard_graphrag_faithfulness": -1,
                "cag_rag_faithfulness": -1,
                "truthful_rag_faithfulness": -1,
                "standard_rag_context": "",
                "standard_graphrag_context": "",
                "cag_rag_context": "",
                "truthful_rag_context": "",
            }
            rows.append(row)

        rate_int = int(rate * 100)
        csv_path = os.path.join(tmpdir, f"sweep_rate{rate_int}.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    # Plot sweep
    saved = plot_injection_sweep(tmpdir, output_dir, prefix="sweep")

    # Single-run bar chart (rate=0.4)
    run_csv = os.path.join(tmpdir, "sweep_rate40.csv")
    saved += plot_metric_comparison(run_csv, output_dir)

    # Faithfulness chart with synthetic scores
    # Re-write rate40 CSV with plausible faithfulness scores
    faith_vals = {
        "Standard RAG":      0.62,
        "TruthfulRAG":       0.71,
        "Standard GraphRAG": 0.55,
        "CAG-RAG":           0.84,
    }
    rows_with_faith = []
    rng2 = random.Random(7)
    with open(run_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            for sys_name, mean_f in faith_vals.items():
                _, _, faith_field = _FIELD_MAP[sys_name]
                row[faith_field] = max(0.0, min(1.0, mean_f + rng2.gauss(0, 0.08)))
            rows_with_faith.append(row)

    faith_csv = os.path.join(tmpdir, "run_with_faith.csv")
    with open(faith_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_with_faith)

    saved += plot_faithfulness(faith_csv, output_dir)

    print(f"\n[Demo] Charts saved to {output_dir}/:")
    for p in saved:
        print(f"  {p}")
    print("\nNote: these use synthetic data. Replace with real results after running:")
    print("  python -m cag_rag.evaluation.run_comparison --sweep --out results/sweep")
    print("  python -m cag_rag.visualization.plot_results --sweep-dir results --out figs")
    return saved


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Plot CAG-RAG evaluation results")
    parser.add_argument("--sweep-dir",  type=str, default="results",
                        help="Directory with sweep CSVs (default: results/)")
    parser.add_argument("--csv",        type=str, default="",
                        help="Single run CSV for bar chart (e.g. results/run1.csv)")
    parser.add_argument("--out",        type=str, default="figs",
                        help="Output directory for PNGs (default: figs/)")
    parser.add_argument("--prefix",     type=str, default="sweep",
                        help="Sweep CSV filename prefix (default: sweep)")
    parser.add_argument("--demo",       action="store_true",
                        help="Generate charts with synthetic demo data (no CSV needed)")
    args = parser.parse_args()

    if args.demo:
        _run_demo(args.out)
    else:
        saved = []

        # Injection-rate sweep chart
        sweep_files = (glob.glob(os.path.join(args.sweep_dir, f"{args.prefix}_rate*.csv")) or
                       glob.glob(os.path.join(args.sweep_dir, "*_rate*.csv")))
        if sweep_files:
            saved += plot_injection_sweep(args.sweep_dir, args.out, args.prefix)
        else:
            print(f"No sweep CSVs found in {args.sweep_dir}/  — skipping sweep chart.")

        # Single-run bar chart
        csv_path = args.csv
        if not csv_path:
            candidates = (glob.glob(os.path.join(args.sweep_dir, "run*.csv")) +
                          glob.glob(os.path.join(args.sweep_dir, "*.csv")))
            candidates = [c for c in candidates if "_rate" not in c]
            if candidates:
                csv_path = candidates[0]

        if csv_path and os.path.isfile(csv_path):
            saved += plot_metric_comparison(csv_path, args.out)
            saved += plot_faithfulness(csv_path, args.out)
        else:
            print("No single-run CSV found — skipping metric comparison chart.")
            print("  Use --csv results/run1.csv to specify one.")

        if saved:
            print(f"\nCharts saved:")
            for p in saved:
                print(f"  {p}")
        else:
            print("No charts generated.")
