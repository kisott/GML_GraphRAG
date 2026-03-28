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

## Implementation Status

### What Has Been Implemented

#### 1. Conflict Detection Module (`cag_rag/conflict/`)
- **SemanticFilter** — filters triple pairs using cosine similarity to find candidate conflict pairs (Stage 1)
- **NLIClassifier** — classifies each candidate pair as contradiction / neutral / entailment using DeBERTa (Stage 2)
- **ConflictDetector** — orchestrates both stages and outputs `ConflictEdge` objects with conflict type, confidence score, and source provenance

#### 2. Knowledge Graph Construction (`cag_rag/graph/`)
- **KnowledgeGraphBuilder** — extracts triples from text documents; supports spaCy (fast) and REBEL (higher quality) backends
- **AugmentedGraphIndex** — builds G' = (V, E ∪ C), storing conflict edges as a separate edge type in a MultiDiGraph with O(1) conflict neighbour lookup

#### 3. Conflict-Aware Retrieval (`cag_rag/retrieval/`)
- **ConflictAwareRetriever** — selects seed nodes via embedding similarity → BFS subgraph expansion → conflict neighbourhood expansion; returns a `RetrievalResult` containing both facts and conflict contexts

#### 4. Contradiction Injection Tool (`cag_rag/injection/`)
- **ConflictInjector** — four injection strategies (cross_entity / temporal / numerical / lexical), automatically selects the best available strategy per triple, logs ground-truth labels
- **match_detected_to_ground_truth()** — computes Precision / Recall / F1 against injected ground truth

#### 5. Generation Module (`cag_rag/generation/`)
- **OpenAIGenerator** — supports `standard` and `conflict_aware` prompt modes with built-in retry logic
- **PromptBuilder** — builds structured prompts; conflict-aware mode includes explicit `[CONFLICT]` blocks
- **Conflict flagging** — detects whether the LLM answer explicitly acknowledges a conflict (Conflict Coverage metric)

#### 6. Baseline Systems (`cag_rag/baselines/`)
- **StandardRAG** — flat embedding similarity retrieval over raw text passages, no graph structure
- **StandardGraphRAG** — graph-based retrieval with no conflict edges, standard prompt

#### 7. Data & Evaluation (`cag_rag/data/`, `cag_rag/evaluation/`)
- **SQuADLoader** — loads SQuAD v1.1 from HuggingFace, grouped by Wikipedia article
- **run_comparison.py** — end-to-end comparison of all three systems; outputs Conflict Coverage and token cost table

---

### What Has NOT Been Implemented Yet

#### High Priority (needed for April evaluation phase)

| Feature | Description |
|---|---|
| **TruthfulRAG baseline** | The most important comparison baseline from the proposal — generation-time conflict resolution via entropy-based filtering. Currently missing. |
| **Faithfulness Score** | LLM-as-judge evaluation measuring whether answers faithfully represent disagreement rather than silently resolving it. Explicitly required in the proposal. |
| **FaithEval / MuSiQue / RealtimeQA loaders** | Only SQuAD is supported. The proposal requires four datasets covering logical, compositional, and temporal conflict scenarios. |

#### Medium Priority

| Feature | Description |
|---|---|
| **Large-scale evaluation script** | Current runs cover only ~9 queries. Need to scale to hundreds and save results to CSV/JSON for analysis. |
| **REBEL triple extraction testing** | The backend exists in the code but has not been tested end-to-end. spaCy extractions are lower quality. |
| **configs/ directory** | The folder exists but is empty. Hyperparameters are currently hardcoded across modules. |

#### Low Priority

| Feature | Description |
|---|---|
| **Graph visualization** | Visualize the conflict edge distribution in G' for the final presentation. |
| **Result caching** | Avoid redundant OpenAI API calls across repeated runs to save cost. |

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
