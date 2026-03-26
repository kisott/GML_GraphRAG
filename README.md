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
├── baselines/       # Standard RAG and Standard GraphRAG baselines
├── data/            # SQuAD dataset loader
└── evaluation/      # Comparison runner (all three systems side-by-side)
```

---

## Setup

### Prerequisites

- Python 3.9+
- Conda (recommended) or virtualenv
- OpenAI API key

### 1. Clone the Repository

```bash
git clone https://github.com/kisott/GML_GraphRAG.git
cd GML_GraphRAG
```

### 2. Create and Activate Environment

```bash
conda create -n cag_rag python=3.10
conda activate cag_rag
```

### 3. Install Dependencies

```bash
pip install -e .
pip install torch networkx sentence-transformers transformers spacy openai datasets
python -m spacy download en_core_web_sm
```

### 4. Set Environment Variables

**Mac / Linux:**
```bash
export OPENAI_API_KEY="sk-your-key-here"
export HF_ENDPOINT="https://hf-mirror.com"   # if HuggingFace is blocked in your region
```

**Windows (PowerShell):**
```powershell
$env:OPENAI_API_KEY="sk-your-key-here"
$env:HF_ENDPOINT="https://hf-mirror.com"
```

---

## Running the Pipeline

### Step 1 — Smoke Test (no API key required)

Verifies the core pipeline: graph construction → conflict detection → conflict-aware retrieval.

```bash
python -m cag_rag.test_pipeline
```

Expected output:
```
--- Detected Conflicts ---
[CONFLICT — temporal] Predicate: 'born_in'
  Source A (wiki_v1): Albert Einstein born_in Ulm
  Source B (wiki_v2): Albert Einstein born_in Munich
  Confidence: 1.00
...
✓ Conflict-aware retrieval surfaced 2 conflict(s).
```

### Step 2 — Conflict Injection Test (no API key required)

Verifies the contradiction injection tool and precision/recall/F1 metrics.

```bash
python -m cag_rag.injection.test_injection
```

Expected output:
```
Precision : 1.0000
Recall    : 1.0000
F1        : 1.0000
```

### Step 3 — Generation Test (requires OpenAI key)

Runs the full pipeline end-to-end with LLM generation.

```bash
python -m cag_rag.generation.test_generation
```

Expected output (conflict-aware mode):
```
Both sources provide conflicting information about his birthplace,
and it is unclear which is correct based on the available data.
Conflict flagged: True
```

### Step 4 — Full Baseline Comparison on SQuAD (requires OpenAI key)

Loads SQuAD articles, injects contradictions, and runs all three systems.

```bash
python -m cag_rag.evaluation.run_comparison
```

Expected output:
```
=================================================================
COMPARISON SUMMARY  (9 queries)
=================================================================
System                  Conflict Coverage   Avg Tokens
-------------------------------------------------------
Standard RAG                        0.0%          553
Standard GraphRAG                   0.0%         1483
CAG-RAG                            44.4%        12812
=================================================================
```

To change the number of articles or questions, edit the `run()` call at the bottom of
[cag_rag/evaluation/run_comparison.py](cag_rag/evaluation/run_comparison.py):

```python
results = run(
    max_articles=5,          # number of Wikipedia articles
    max_qa_per_article=3,    # questions per article
    injection_rate=0.4,      # fraction of triples to inject conflicts into
)
```

---

## System Overview

Three systems are compared under the same injected-conflict conditions:

| System | Retrieval | Conflict Handling | Prompt |
|---|---|---|---|
| Standard RAG | Embedding similarity over raw passages | None | Standard |
| Standard GraphRAG | BFS subgraph traversal, no conflict edges | None | Standard |
| CAG-RAG | BFS subgraph + conflict neighbourhood expansion | Index-time conflict edges | Conflict-aware |

### Key Metrics

| Metric | Description |
|---|---|
| Conflict Coverage | % of answers that explicitly acknowledge a conflict |
| Precision / Recall / F1 | Conflict detection accuracy vs. injected ground-truth labels |
| Avg Tokens | Proxy for cost and latency |

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
