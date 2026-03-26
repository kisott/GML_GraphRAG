"""
Knowledge Graph Builder
-----------------------
Converts raw text documents into a NetworkX DiGraph of (subject, predicate, object)
triples, which then feeds into the augmented index.

Two extraction backends are supported:
  1. "spacy"  — rule-based NER + dependency parsing (fast, no GPU needed)
  2. "rebel"  — REBEL relation extraction model (higher recall, needs transformers)

The builder outputs:
  - A list of Triple objects (for the conflict detector)
  - A NetworkX DiGraph G = (V, E) where:
      nodes = entities  (label, entity_type, source_docs)
      edges = relations (predicate, source_doc, triple_id)
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Tuple

import networkx as nx

from ..conflict.semantic_filter import Triple

logger = logging.getLogger(__name__)

ExtractionBackend = Literal["spacy", "rebel"]


# ---------------------------------------------------------------------------
# Document input format
# ---------------------------------------------------------------------------

@dataclass
class Document:
    """A single text document (chunk) from the corpus."""
    doc_id: str
    text: str
    metadata: Dict = field(default_factory=dict)   # e.g. {"title": ..., "url": ...}


# ---------------------------------------------------------------------------
# Node / Edge ID helpers
# ---------------------------------------------------------------------------

def _entity_id(label: str) -> str:
    """Stable node ID: lowercase, strip whitespace, replace spaces with '_'."""
    return label.strip().lower().replace(" ", "_")


def _triple_id(subject: str, predicate: str, obj: str) -> str:
    key = f"{subject}||{predicate}||{obj}"
    return hashlib.md5(key.encode()).hexdigest()[:12]


# ---------------------------------------------------------------------------
# spaCy extraction (rule-based, fast)
# ---------------------------------------------------------------------------

def _extract_spacy(doc: Document, nlp) -> List[Triple]:
    """
    Lightweight relation extraction using spaCy:
      - Detects named entities (NER)
      - Connects entity pairs that co-occur in the same sentence via
        the root verb / nominal head as the predicate.
    This is intentionally simple — replace with REBEL for higher quality.
    """
    triples: List[Triple] = []
    parsed = nlp(doc.text)

    for sent in parsed.sents:
        ents = list(sent.ents)
        if len(ents) < 2:
            continue

        # Find the root verb token of the sentence as the predicate proxy
        root = next((t for t in sent if t.dep_ == "ROOT"), None)
        predicate = root.lemma_ if root else "related_to"

        # Connect consecutive entity pairs with the predicate
        for i in range(len(ents) - 1):
            subj_ent = ents[i]
            obj_ent = ents[i + 1]
            t = Triple(
                subject=subj_ent.text,
                predicate=predicate,
                obj=obj_ent.text,
                source_doc=doc.doc_id,
                triple_id=_triple_id(subj_ent.text, predicate, obj_ent.text),
            )
            triples.append(t)

    return triples


# ---------------------------------------------------------------------------
# REBEL extraction (transformer-based, higher quality)
# ---------------------------------------------------------------------------

def _extract_rebel(doc: Document, tokenizer, model) -> List[Triple]:
    """
    Use REBEL (Cabot & Navigli, 2021) to extract (head, relation, tail) triples.
    Model: Babelscape/rebel-large on HuggingFace.
    """
    import torch

    # REBEL processes text in chunks of ~128 tokens
    inputs = tokenizer(
        doc.text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    )
    with torch.no_grad():
        generated = model.generate(
            **inputs,
            num_beams=3,
            max_length=256,
        )
    decoded = tokenizer.batch_decode(generated, skip_special_tokens=False)[0]

    triples = _parse_rebel_output(decoded, doc.doc_id)
    return triples


def _parse_rebel_output(text: str, source_doc: str) -> List[Triple]:
    """Parse REBEL's special-token output format into Triple objects."""
    triples = []
    current = {}
    # REBEL output format: <triplet> head <subj> relation <obj> tail
    for token in text.split("<"):
        token = token.strip()
        if token.startswith("triplet>"):
            if current:
                triples.append(_rebel_dict_to_triple(current, source_doc))
            current = {}
        elif token.startswith("subj>"):
            current["predicate"] = token[len("subj>"):].strip()
        elif token.startswith("obj>"):
            current["obj"] = token[len("obj>"):].strip()
        elif token.startswith("triplet>") or ">" not in token:
            if "subject" not in current:
                current["subject"] = token.replace(">", "").strip()
    if current:
        triples.append(_rebel_dict_to_triple(current, source_doc))
    return [t for t in triples if t is not None]


def _rebel_dict_to_triple(d: dict, source_doc: str) -> Optional[Triple]:
    try:
        return Triple(
            subject=d["subject"],
            predicate=d["predicate"],
            obj=d["obj"],
            source_doc=source_doc,
            triple_id=_triple_id(d["subject"], d["predicate"], d["obj"]),
        )
    except KeyError:
        return None


# ---------------------------------------------------------------------------
# KnowledgeGraphBuilder
# ---------------------------------------------------------------------------

class KnowledgeGraphBuilder:
    """
    Builds a NetworkX DiGraph from a list of Documents.

    Node attributes:
        label       : entity surface form
        entity_type : NER type (PERSON, ORG, …) — spaCy only
        source_docs : set of doc_ids mentioning this entity

    Edge attributes:
        predicate   : relation label
        source_doc  : document where this triple was extracted
        triple_id   : deterministic hash ID (used for dedup)
        weight      : co-occurrence count (incremented on duplicate triples)
    """

    def __init__(
        self,
        backend: ExtractionBackend = "spacy",
        spacy_model: str = "en_core_web_sm",
        rebel_model: str = "Babelscape/rebel-large",
        device: str = "cpu",
    ):
        self.backend = backend
        self.spacy_model = spacy_model
        self.rebel_model = rebel_model
        self.device = device
        self._nlp = None
        self._rebel_tokenizer = None
        self._rebel_model = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(self, documents: List[Document]) -> Tuple[nx.DiGraph, List[Triple]]:
        """
        Process all documents, extract triples, and return:
          - G : base knowledge graph (NetworkX DiGraph)
          - all_triples : flat list of Triple objects (input to conflict detector)
        """
        self._load_backend()

        all_triples: List[Triple] = []
        for doc in documents:
            logger.info(f"Extracting triples from doc '{doc.doc_id}' …")
            if self.backend == "spacy":
                triples = _extract_spacy(doc, self._nlp)
            else:
                triples = _extract_rebel(doc, self._rebel_tokenizer, self._rebel_model)
            logger.info(f"  → {len(triples)} triples extracted")
            all_triples.extend(triples)

        G = self._triples_to_graph(all_triples)
        logger.info(
            f"Knowledge graph built: {G.number_of_nodes()} nodes, "
            f"{G.number_of_edges()} edges, from {len(all_triples)} triples"
        )
        return G, all_triples

    def build_from_triples(self, triples: List[Triple]) -> nx.DiGraph:
        """Build a graph directly from an existing list of Triple objects."""
        return self._triples_to_graph(triples)

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------

    @staticmethod
    def _triples_to_graph(triples: List[Triple]) -> nx.DiGraph:
        G = nx.DiGraph()
        seen_triples: Dict[str, int] = {}   # triple_id → edge index (for weight)

        for t in triples:
            subj_id = _entity_id(t.subject)
            obj_id = _entity_id(t.obj)

            # Add / update subject node
            if subj_id not in G:
                G.add_node(subj_id, label=t.subject, source_docs=set(), entity_type="")
            G.nodes[subj_id]["source_docs"].add(t.source_doc)

            # Add / update object node
            if obj_id not in G:
                G.add_node(obj_id, label=t.obj, source_docs=set(), entity_type="")
            G.nodes[obj_id]["source_docs"].add(t.source_doc)

            # Add edge (increment weight if duplicate triple_id)
            tid = t.triple_id or _triple_id(t.subject, t.predicate, t.obj)
            if G.has_edge(subj_id, obj_id) and G[subj_id][obj_id].get("triple_id") == tid:
                G[subj_id][obj_id]["weight"] += 1
            else:
                G.add_edge(
                    subj_id, obj_id,
                    predicate=t.predicate,
                    source_doc=t.source_doc,
                    triple_id=tid,
                    edge_type="relation",   # distinguishes from conflict edges in G'
                    weight=1,
                )

        return G

    # ------------------------------------------------------------------
    # Lazy backend loading
    # ------------------------------------------------------------------

    def _load_backend(self):
        if self.backend == "spacy":
            if self._nlp is None:
                try:
                    import spacy
                    self._nlp = spacy.load(self.spacy_model)
                except OSError:
                    raise OSError(
                        f"spaCy model '{self.spacy_model}' not found. "
                        f"Run: python -m spacy download {self.spacy_model}"
                    )
        elif self.backend == "rebel":
            if self._rebel_tokenizer is None:
                from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
                self._rebel_tokenizer = AutoTokenizer.from_pretrained(self.rebel_model)
                self._rebel_model = AutoModelForSeq2SeqLM.from_pretrained(self.rebel_model)
                if self.device == "cuda":
                    self._rebel_model = self._rebel_model.cuda()
