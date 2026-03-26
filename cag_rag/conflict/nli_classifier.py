"""
Stage 2: NLI Conflict Classifier
----------------------------------
Takes candidate pairs from SemanticFilter and classifies each pair as:
  - "contradiction"  → conflict edge should be added to G'
  - "neutral"        → no structural relationship, discard
  - "entailment"     → paraphrase / consistent, discard

Default backbone: cross-encoder/nli-deberta-v3-large
  - Fine-tuned on MNLI/SNLI for 3-class NLI
  - DeBERTa architecture (reference [12] in the proposal)

The classifier takes the *object texts* of the two triples as
(premise, hypothesis) because the subject+predicate context is the
same for both — only the object values conflict.
Optionally, full triple text can be used as premise/hypothesis.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Literal, Tuple

logger = logging.getLogger(__name__)

# Label order returned by cross-encoder NLI models (model-dependent)
# For cross-encoder/nli-deberta-v3-large the order is:
#   0 → contradiction, 1 → entailment, 2 → neutral
_LABEL_ORDER = {
    "cross-encoder/nli-deberta-v3-large": ["contradiction", "entailment", "neutral"],
    "cross-encoder/nli-deberta-v3-base":  ["contradiction", "entailment", "neutral"],
    # HuggingFace zero-shot pipeline label map (for fallback)
    "typeform/distilbert-base-uncased-mnli": ["entailment", "neutral", "contradiction"],
}

NLILabel = Literal["contradiction", "entailment", "neutral"]


@dataclass
class NLIResult:
    premise: str
    hypothesis: str
    label: NLILabel
    contradiction_prob: float
    entailment_prob: float
    neutral_prob: float


class NLIClassifier:
    """
    Wraps a cross-encoder NLI model to classify (premise, hypothesis) pairs.

    Two inference modes:
      - "cross_encoder"  : uses sentence-transformers CrossEncoder (faster batching)
      - "pipeline"       : uses HuggingFace zero-shot-classification pipeline (simpler)

    For production use, prefer "cross_encoder" with deberta-v3-large.
    For quick testing without GPU, "pipeline" with a smaller model is fine.
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/nli-deberta-v3-large",
        mode: Literal["cross_encoder", "pipeline"] = "cross_encoder",
        contradiction_threshold: float = 0.5,
        device: str = "cpu",
        batch_size: int = 16,
    ):
        self.model_name = model_name
        self.mode = mode
        self.contradiction_threshold = contradiction_threshold
        self.device = device
        self.batch_size = batch_size
        self._model = None

    # ------------------------------------------------------------------
    # Lazy model loading
    # ------------------------------------------------------------------

    def _load(self):
        if self.mode == "cross_encoder":
            try:
                from sentence_transformers import CrossEncoder
            except ImportError as e:
                raise ImportError(
                    "sentence-transformers is required. "
                    "Install with: pip install sentence-transformers"
                ) from e
            logger.info(f"Loading CrossEncoder: {self.model_name}")
            self._model = CrossEncoder(self.model_name, device=self.device)
        else:
            try:
                from transformers import pipeline
            except ImportError as e:
                raise ImportError(
                    "transformers is required. "
                    "Install with: pip install transformers"
                ) from e
            logger.info(f"Loading zero-shot pipeline: {self.model_name}")
            self._model = pipeline(
                "zero-shot-classification",
                model=self.model_name,
                device=0 if self.device == "cuda" else -1,
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def classify_pairs(
        self,
        pairs: List[Tuple[str, str]],   # list of (premise, hypothesis) strings
    ) -> List[NLIResult]:
        """
        Classify a list of (premise, hypothesis) string pairs.
        Returns one NLIResult per pair.
        """
        if not pairs:
            return []
        if self._model is None:
            self._load()

        if self.mode == "cross_encoder":
            return self._classify_cross_encoder(pairs)
        else:
            return self._classify_pipeline(pairs)

    def is_contradiction(self, premise: str, hypothesis: str) -> Tuple[bool, float]:
        """
        Convenience method: returns (is_contradiction, contradiction_probability).
        """
        result = self.classify_pairs([(premise, hypothesis)])[0]
        return result.contradiction_prob >= self.contradiction_threshold, result.contradiction_prob

    # ------------------------------------------------------------------
    # Backend implementations
    # ------------------------------------------------------------------

    def _classify_cross_encoder(self, pairs: List[Tuple[str, str]]) -> List[NLIResult]:
        label_order = _LABEL_ORDER.get(
            self.model_name,
            ["contradiction", "entailment", "neutral"]   # sensible default
        )

        import numpy as np
        from scipy.special import softmax  # ships with scipy

        results = []
        # CrossEncoder.predict returns raw logits; batch for efficiency
        for start in range(0, len(pairs), self.batch_size):
            batch = pairs[start : start + self.batch_size]
            logits = self._model.predict(batch)   # shape: (batch, 3)
            if logits.ndim == 1:
                logits = logits[np.newaxis, :]
            probs = softmax(logits, axis=1)

            for i, (premise, hyp) in enumerate(batch):
                prob_map = {label: float(probs[i, k]) for k, label in enumerate(label_order)}
                label = max(prob_map, key=prob_map.__getitem__)
                results.append(NLIResult(
                    premise=premise,
                    hypothesis=hyp,
                    label=label,
                    contradiction_prob=prob_map.get("contradiction", 0.0),
                    entailment_prob=prob_map.get("entailment", 0.0),
                    neutral_prob=prob_map.get("neutral", 0.0),
                ))
        return results

    def _classify_pipeline(self, pairs: List[Tuple[str, str]]) -> List[NLIResult]:
        candidate_labels = ["contradiction", "entailment", "neutral"]
        results = []
        for premise, hyp in pairs:
            out = self._model(
                premise,
                candidate_labels=candidate_labels,
                hypothesis_template="{}",
                multi_label=False,
            )
            # pipeline returns labels sorted by score descending
            prob_map = dict(zip(out["labels"], out["scores"]))
            label = out["labels"][0]
            results.append(NLIResult(
                premise=premise,
                hypothesis=hyp,
                label=label,
                contradiction_prob=prob_map.get("contradiction", 0.0),
                entailment_prob=prob_map.get("entailment", 0.0),
                neutral_prob=prob_map.get("neutral", 0.0),
            ))
        return results
