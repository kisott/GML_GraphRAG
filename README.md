# CAG-RAG: Conflict-Aware GraphRAG

**ECE 6502: Machine Learning on Graphs — Spring 2026, University of Virginia**

> Tammy Ngo, Xian Wu, West Plowman, Qizhen Yang, Maddie Priebe

CAG-RAG is a Conflict-Aware Graph Retrieval-Augmented Generation framework that encodes contradictions as explicit typed edges in the knowledge graph index at construction time, and retrieves conflict neighbourhoods alongside relevant subgraphs so the LLM receives explicitly structured disagreement rather than silently contradictory context.

---

## Project Structure

```
cag_rag/
├── conflict/        # Stage 1 semantic filter + Stage 2 NLI classifier
├── graph/           # Knowledge graph builder + augmented index G' = (V, E ∪ C)
├── retrieval/       # Conflict-aware retriever (BFS + conflict neighbourhood expansion)
├── injection/       # Contradiction injection tool + precision/recall/F1 evaluation
├── generation/      # OpenAI generation module (standard / conflict-aware modes)
├── baselines/       # Standard RAG, Standard GraphRAG, and TruthfulRAG baselines
├── data/            # SQuAD dataset loader
├── evaluation/      # Comparison runner + LLM-as-judge faithfulness scorer
└── visualization/   # Graph plots + result analysis charts
```

---

## Complete Workflow (Start Here)

This section walks through every step from environment setup to generating final charts.
Follow the steps in order the first time.

### Step 0 — Clone and Set Up Environment

```bash
git clone https://github.com/kisott/GML_GraphRAG.git
cd GML_GraphRAG

conda create -n cag_rag python=3.10
conda activate cag_rag

pip install -e .
pip install torch networkx sentence-transformers transformers spacy openai datasets scipy matplotlib
python -m spacy download en_core_web_sm
```

### Step 1 — Set Your API Key

You need an OpenAI API key for any step that calls the LLM (Steps 4 onward).

**Mac / Linux (bash):**
```bash
export OPENAI_API_KEY="sk-your-key-here"
export HF_ENDPOINT="https://hf-mirror.com"   # add this if HuggingFace is blocked
```

**Windows (PowerShell):**
```powershell
$env:OPENAI_API_KEY = "sk-your-key-here"
$env:HF_ENDPOINT = "https://hf-mirror.com"
```

> The key must be set every time you open a new terminal. It is **not** saved between sessions.

---

### Step 2 — Verify the Core Pipeline (no API key needed)

Run the smoke test to confirm graph building, conflict detection, and retrieval all work:

```bash
python -m cag_rag.test_pipeline
```

Expected output (last lines):
```
--- Detected Conflicts ---
[CONFLICT — temporal] Predicate: 'born_in'
  Source A (wiki_v1): Albert Einstein born_in Ulm
  Source B (wiki_v2): Albert Einstein born_in Munich
  Confidence: 1.00
✓ Conflict-aware retrieval surfaced 2 conflict(s).
```

If this fails, check that spaCy and sentence-transformers are installed correctly.

---

### Step 3 — Verify the Injection Tool (no API key needed)

Confirms that synthetic contradictions are injected and detected at 100% precision/recall:

```bash
python -m cag_rag.injection.test_injection
```

Expected output:
```
Precision : 1.0000
Recall    : 1.0000
F1        : 1.0000
```

---

### Step 4 — Verify Generation (requires OpenAI key)

Tests the LLM generation module end-to-end with a small hardcoded example:

```bash
python -m cag_rag.generation.test_generation
```

Expected output (conflict-aware mode):
```
Both sources provide conflicting information about his birthplace,
and it is unclear which is correct based on the available data.
Conflict flagged: True
```

If you see `ValueError: Set OPENAI_API_KEY`, go back to Step 1.

---

### Step 5 — Quick Sanity Check on SQuAD (requires OpenAI key)

Run a small evaluation (3 articles × 2 QA = 6 queries) to make sure the full pipeline works before committing to the full run:

```bash
python -m cag_rag.evaluation.run_comparison --articles 3 --qa 2
```

Takes ~5 minutes. If this completes without errors, you are ready for the full evaluation.

---

### Step 6 — Full Evaluation on SQuAD (requires OpenAI key)

Runs all four systems on 50 queries and saves results to CSV/JSON.
**Takes ~25 minutes on CPU.**

```bash
python -m cag_rag.evaluation.run_comparison --out results/run1
```

Output files:
- `results/run1.csv` — per-query results for all systems
- `results/run1.json` — same data in JSON format

To also score faithfulness (LLM-as-judge, adds ~10 more minutes):
```bash
python -m cag_rag.evaluation.run_comparison --faithfulness --out results/run1
```

---

### Step 7 — Injection-Rate Sweep Experiment (requires OpenAI key)

Runs the comparison at four injection rates (0%, 20%, 40%, 60%) to show how conflict density affects each system.
**Takes ~1.5 hours on CPU.**

```bash
python -m cag_rag.evaluation.run_comparison --sweep --out results/sweep
```

Output files:
- `results/sweep_rate0.csv`
- `results/sweep_rate20.csv`
- `results/sweep_rate40.csv`
- `results/sweep_rate60.csv`

---

### Step 8 — Generate Charts

**Option A — Use demo data (no evaluation needed, instant):**
```bash
python -m cag_rag.visualization.plot_results --demo --out figs
```

**Option B — Use your real evaluation results:**
```bash
# After Step 6 (single run bar chart)
python -m cag_rag.visualization.plot_results --csv results/run1.csv --out figs

# After Step 7 (sweep line chart + bar chart)
python -m cag_rag.visualization.plot_results --sweep-dir results --out figs
```

Output files in `figs/`:
- `sweep_line_chart.png` — injection rate vs. F1 and Conflict Coverage (line chart)
- `metric_comparison_bar.png` — all four systems × all metrics (grouped bar chart)
- `faithfulness_bar.png` — LLM-as-judge scores per system (if faithfulness was run)

---

### Step 9 — Visualize the Knowledge Graph

Produces a graph diagram of G' showing relation edges (gray) and conflict edges (red dashed).
No API key required. Uses a small built-in demo dataset.

```bash
python -m cag_rag.visualization.graph_viz
```

Output files in `figs/`:
- `augmented_graph.png` — full G' with all nodes and edges
- `conflict_subgraph.png` — zoomed view of conflict neighbourhood only

---

## All CLI Options Reference

### `run_comparison.py`

```bash
python -m cag_rag.evaluation.run_comparison [options]
```

| Flag | Default | Description |
|---|---|---|
| `--articles N` | 10 | Number of SQuAD articles to load |
| `--qa N` | 5 | QA pairs per article (total queries = articles × qa) |
| `--rate F` | 0.4 | Conflict injection rate for a single run (0.0–1.0) |
| `--top-k N` | 3 | Number of passages / nodes to retrieve |
| `--sweep` | off | Run at injection rates 0%, 20%, 40%, 60% |
| `--no-truthful` | off | Skip TruthfulRAG baseline (faster) |
| `--faithfulness` | off | Run LLM-as-judge faithfulness scoring (extra API calls) |
| `--out PREFIX` | — | Save results: `PREFIX.csv` and `PREFIX.json` |

### `plot_results.py`

```bash
python -m cag_rag.visualization.plot_results [options]
```

| Flag | Default | Description |
|---|---|---|
| `--demo` | off | Generate charts with synthetic data, no CSV needed |
| `--sweep-dir DIR` | `results` | Directory containing sweep CSV files |
| `--csv PATH` | auto | Single run CSV for bar chart |
| `--prefix STR` | `sweep` | Filename prefix for sweep CSVs |
| `--out DIR` | `figs` | Output directory for PNG files |

---

## System Overview

Four systems are compared under the same injected-conflict conditions:

| System | Retrieval | Conflict Handling | Prompt |
|---|---|---|---|
| Standard RAG | Embedding similarity over raw passages | None | Standard |
| TruthfulRAG | Embedding + NLI consistency filtering | Filters conflicting passages at retrieval time | Standard |
| Standard GraphRAG | BFS subgraph traversal, no conflict edges | None | Standard |
| CAG-RAG | BFS subgraph + conflict neighbourhood expansion | Index-time conflict edges | Conflict-aware |

### Key Metrics

| Metric | Description |
|---|---|
| Exact Match (EM) | % of answers matching the gold span after normalization |
| Token-level F1 | Average token overlap with gold answer (SQuAD-standard) |
| Conflict Coverage | % of answers that explicitly acknowledge a conflict |
| Faithfulness Score | LLM-as-judge: factual grounding + conflict acknowledgment (0–1) |
| Avg Tokens | Proxy for cost and latency |

> **Note on EM/F1:** SQuAD gold answers are short extractive spans. Generative models produce full sentences, so F1 scores appear lower than for extractive systems even when factually correct. **Conflict Coverage and Faithfulness Score are the primary metrics** for evaluating CAG-RAG's contribution.

---

## Troubleshooting

| Error | Cause | Fix |
|---|---|---|
| `ValueError: Set OPENAI_API_KEY` | API key not set | Run `export OPENAI_API_KEY="sk-..."` in the same terminal |
| `No module named 'spacy'` | Missing dependency | `pip install spacy && python -m spacy download en_core_web_sm` |
| `No module named 'sentence_transformers'` | Missing dependency | `pip install sentence-transformers` |
| `FileNotFoundError: results/sweep_rate0.csv` | results/ dir missing | Fixed automatically — just re-run |
| `No sweep CSVs found` | Step 7 not run yet | Run `--sweep --out results/sweep` first, or use `--demo` |
| HuggingFace download fails | Network blocked | Set `export HF_ENDPOINT="https://hf-mirror.com"` |
| Very slow NLI inference | Running on CPU | Expected — DeBERTa is slow on CPU. GPU speeds it up 5–10× |

---

## Implementation Status

### Implemented

#### 1. Conflict Detection (`cag_rag/conflict/`)
- **SemanticFilter** — cosine similarity filtering of triple pairs (Stage 1)
- **NLIClassifier** — DeBERTa-based 3-class NLI classifier: contradiction / neutral / entailment (Stage 2)
- **ConflictDetector** — orchestrates both stages, outputs `ConflictEdge` objects with type, confidence, and provenance

#### 2. Knowledge Graph Construction (`cag_rag/graph/`)
- **KnowledgeGraphBuilder** — triple extraction from text; spaCy (fast) and REBEL (higher quality) backends
- **AugmentedGraphIndex** — builds G' = (V, E ∪ C), MultiDiGraph with O(1) conflict neighbour lookup

#### 3. Conflict-Aware Retrieval (`cag_rag/retrieval/`)
- **ConflictAwareRetriever** — embedding seed → BFS expansion → conflict neighbourhood expansion

#### 4. Contradiction Injection (`cag_rag/injection/`)
- **ConflictInjector** — four strategies: cross_entity / temporal / numerical / lexical
- **match_detected_to_ground_truth()** — Precision / Recall / F1 against injected ground truth

#### 5. Generation (`cag_rag/generation/`)
- **OpenAIGenerator** — `standard` and `conflict_aware` prompt modes, retry logic
- **PromptBuilder** — structured prompts with explicit `[CONFLICT]` blocks

#### 6. Baselines (`cag_rag/baselines/`)
- **StandardRAG** — flat embedding retrieval over raw passages
- **StandardGraphRAG** — graph BFS retrieval, no conflict edges
- **TruthfulRAG** — NLI consistency-filtered retrieval (entropy-based conflict suppression)

#### 7. Data & Evaluation (`cag_rag/data/`, `cag_rag/evaluation/`)
- **SQuADLoader** — SQuAD v1.1 from HuggingFace, grouped by Wikipedia article
- **run_comparison.py** — end-to-end comparison of all four systems; EM, F1, Conflict Coverage, token cost; injection-rate sweep; CSV/JSON export
- **faithfulness.py** — LLM-as-judge faithfulness scorer (factual grounding + conflict acknowledgment)

#### 8. Visualization (`cag_rag/visualization/`)
- **graph_viz.py** — plots G' with relation/conflict edges; conflict subgraph zoom view
- **plot_results.py** — injection-rate sweep line chart, metric comparison bar chart, faithfulness bar chart; demo mode with synthetic data

---

### Still To Do

| Feature | Priority | Description |
|---|---|---|
| Additional dataset loaders | High | FaithEval, MuSiQue, RealtimeQA — only SQuAD currently supported |
| REBEL triple extraction testing | Medium | Backend exists but untested end-to-end; spaCy quality is lower |
| Result caching | Low | Avoid redundant OpenAI API calls across repeated runs |

---

## Pipeline Architecture

```
Documents
    │
    ▼
KnowledgeGraphBuilder          ← spaCy or REBEL extraction
    │  G = (V, E)
    ▼
ConflictInjector               ← injects synthetic contradictions (evaluation only)
    │  augmented triples + ground-truth labels
    ▼
ConflictDetector               ← SemanticFilter (Stage 1) + NLI classifier (Stage 2)
    │  ConflictEdge list
    ▼
AugmentedGraphIndex            ← G' = (V, E ∪ C)  heterogeneous graph
    │
    ▼
ConflictAwareRetriever         ← BFS on relation edges + conflict neighbourhood expansion
    │  RetrievalResult (facts + conflict contexts)
    ▼
OpenAIGenerator                ← standard or conflict-aware prompt
    │
    ▼
Answer  [CONFLICT FLAGGED if disagreement is surfaced]
```
