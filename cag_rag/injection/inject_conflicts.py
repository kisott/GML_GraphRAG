"""
Contradiction Injection Tool
-----------------------------
Generates controlled synthetic contradictions into a list of Triple objects.
Produces ground-truth labels used to evaluate conflict detection precision/recall/F1.

Injection strategies
--------------------
cross_entity  — Swap the object of one entity with another entity's object for the
                same predicate.  e.g. Einstein born_in Ulm  →  Einstein born_in Warsaw
                (Warsaw is drawn from Marie Curie's born_in triple in the corpus).
                Most realistic: the injected value always comes from the real corpus.

numerical     — Perturb numeric substrings in the object.
                e.g. "population 3400000"  →  "population 3841200"

temporal      — Shift 4-digit years by a small random offset (±2–9 years).
                e.g. "born_in 1879"  →  "born_in 1883"

lexical       — Replace the whole object with a near-opposite from a small lookup
                table of common values.
                e.g. "German"  →  "Austrian"

Usage
-----
    from cag_rag.conflict.semantic_filter import Triple
    from cag_rag.injection import ConflictInjector

    injector = ConflictInjector(injection_rate=0.3, seed=42)
    result = injector.inject(triples)

    # Augmented triples (original + injected contradictions)
    augmented = result.augmented_triples

    # Ground-truth records for evaluation
    for gt in result.ground_truth:
        print(gt)

Evaluation helpers
------------------
    from cag_rag.injection.inject_conflicts import match_detected_to_ground_truth
    metrics = match_detected_to_ground_truth(result.ground_truth, detected_conflict_edges)
"""

from __future__ import annotations

import logging
import random
import re
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Tuple

from ..conflict.semantic_filter import Triple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Injection type
# ---------------------------------------------------------------------------

InjectionType = Literal["cross_entity", "numerical", "temporal", "lexical"]

# ---------------------------------------------------------------------------
# Small lexical substitution table
# Near-opposite or plausible-but-wrong replacements for frequent object values
# ---------------------------------------------------------------------------

_LEXICAL_TABLE: Dict[str, List[str]] = {
    # nationalities / demonyms
    "german":     ["austrian", "swiss", "french", "polish"],
    "austrian":   ["german", "swiss", "hungarian"],
    "polish":     ["german", "french", "russian"],
    "french":     ["german", "belgian", "swiss"],
    "british":    ["american", "australian", "irish"],
    "american":   ["british", "canadian", "australian"],
    # cities
    "ulm":        ["munich", "berlin", "frankfurt", "stuttgart"],
    "munich":     ["ulm", "berlin", "hamburg", "vienna"],
    "berlin":     ["munich", "hamburg", "cologne"],
    "warsaw":     ["krakow", "lodz", "poznan"],
    "paris":      ["lyon", "marseille", "bordeaux"],
    "london":     ["manchester", "birmingham", "glasgow"],
    "new york":   ["chicago", "los angeles", "boston"],
    "princeton":  ["harvard", "yale", "columbia", "stanford"],
    # prizes / awards
    "nobel prize in physics":    ["nobel prize in chemistry", "nobel prize in medicine"],
    "nobel prize in chemistry":  ["nobel prize in physics", "nobel prize in medicine"],
    "nobel prize in medicine":   ["nobel prize in physics", "nobel prize in chemistry"],
    # institutions
    "eth zurich": ["university of zurich", "epfl", "heidelberg university"],
    "harvard":    ["yale", "princeton", "columbia", "stanford", "mit"],
    "mit":        ["harvard", "caltech", "stanford"],
    # generic truth values
    "true":  ["false"],
    "false": ["true"],
    "yes":   ["no"],
    "no":    ["yes"],
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class InjectedConflict:
    """
    Ground-truth record for one injected contradiction.

    Used to evaluate whether the conflict detector found this conflict:
      - match on (subject, predicate) to link to a detected ConflictEdge
    """
    conflict_id: str                  # unique identifier
    original_triple: Triple           # the real triple that was kept
    injected_triple: Triple           # the synthetic contradicting triple
    injection_type: InjectionType
    conflicting_predicate: str        # always == original_triple.predicate
    source_original: str              # source_doc of the original triple
    source_injected: str              # source_doc of the injected triple (synthetic ID)

    def __repr__(self) -> str:
        return (
            f"[{self.injection_type}] "
            f"'{self.original_triple.subject}' {self.conflicting_predicate}: "
            f"'{self.original_triple.obj}' (real) "
            f"vs '{self.injected_triple.obj}' (injected)"
        )

    @property
    def subject(self) -> str:
        return self.original_triple.subject

    @property
    def predicate(self) -> str:
        return self.conflicting_predicate


@dataclass
class InjectionResult:
    """
    Output of ConflictInjector.inject().

    Attributes
    ----------
    augmented_triples : List[Triple]
        Original triples + injected contradicting triples.
        Pass this to KnowledgeGraphBuilder / ConflictDetector.
    ground_truth : List[InjectedConflict]
        One record per injected conflict, used to compute precision/recall/F1.
    n_original : int
        Number of triples before injection.
    n_injected : int
        Number of synthetic contradictions added.
    """
    augmented_triples: List[Triple]
    ground_truth: List[InjectedConflict]
    n_original: int
    n_injected: int

    def summary(self) -> str:
        type_counts: Dict[str, int] = defaultdict(int)
        for gt in self.ground_truth:
            type_counts[gt.injection_type] += 1
        lines = [
            f"InjectionResult: {self.n_original} original triples, "
            f"{self.n_injected} injected conflicts",
        ]
        for itype, cnt in sorted(type_counts.items()):
            lines.append(f"  {itype:15s}: {cnt}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# ConflictInjector
# ---------------------------------------------------------------------------

class ConflictInjector:
    """
    Injects synthetic contradictions into a list of Triple objects.

    Parameters
    ----------
    injection_rate : float
        Fraction of triples to inject a contradiction for (0–1).
        Default 0.3 → ~30 % of triples get an injected counterpart.
    strategy : str | list[str]
        Which injection strategies to use.  "auto" selects the best available
        strategy per triple (cross_entity → temporal → numerical → lexical).
        Or pass a list like ["cross_entity", "temporal"] to restrict choices.
    injected_source_prefix : str
        Source-doc ID prefix for injected triples.  The conflict_id is appended.
    seed : int | None
        Random seed for reproducibility.
    """

    def __init__(
        self,
        injection_rate: float = 0.3,
        strategy: str | List[str] = "auto",
        injected_source_prefix: str = "injected",
        seed: Optional[int] = 42,
    ):
        self.injection_rate = injection_rate
        self.strategy = [strategy] if isinstance(strategy, str) else strategy
        self.injected_source_prefix = injected_source_prefix
        self._rng = random.Random(seed)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def inject(self, triples: List[Triple]) -> InjectionResult:
        """
        Inject contradictions into a subset of the given triples.

        Returns an InjectionResult with both the augmented triple list
        and ground-truth conflict records.
        """
        if not triples:
            return InjectionResult(
                augmented_triples=[],
                ground_truth=[],
                n_original=0,
                n_injected=0,
            )

        # Pre-build predicate → triples index for cross-entity strategy
        pred_index: Dict[str, List[Triple]] = defaultdict(list)
        for t in triples:
            pred_index[t.predicate.strip().lower()].append(t)

        # Sample which triples to target
        n_target = max(1, round(len(triples) * self.injection_rate))
        targets = self._rng.sample(triples, min(n_target, len(triples)))

        injected_triples: List[Triple] = []
        ground_truth: List[InjectedConflict] = []

        for original in targets:
            result = self._inject_one(original, pred_index)
            if result is None:
                continue
            injected_triple, itype = result
            cid = str(uuid.uuid4())[:8]
            injected_triple = Triple(
                subject=injected_triple.subject,
                predicate=injected_triple.predicate,
                obj=injected_triple.obj,
                source_doc=f"{self.injected_source_prefix}_{cid}",
                triple_id=f"inj_{cid}",
            )
            gt = InjectedConflict(
                conflict_id=cid,
                original_triple=original,
                injected_triple=injected_triple,
                injection_type=itype,
                conflicting_predicate=original.predicate,
                source_original=original.source_doc,
                source_injected=injected_triple.source_doc,
            )
            injected_triples.append(injected_triple)
            ground_truth.append(gt)
            logger.debug(f"Injected: {gt}")

        augmented = list(triples) + injected_triples
        logger.info(
            f"Injection complete: {len(triples)} original, "
            f"{len(injected_triples)} injected"
        )
        return InjectionResult(
            augmented_triples=augmented,
            ground_truth=ground_truth,
            n_original=len(triples),
            n_injected=len(injected_triples),
        )

    # ------------------------------------------------------------------
    # Strategy dispatcher
    # ------------------------------------------------------------------

    def _inject_one(
        self,
        triple: Triple,
        pred_index: Dict[str, List[Triple]],
    ) -> Optional[Tuple[Triple, InjectionType]]:
        """
        Try each enabled strategy in priority order.
        Returns (injected_triple, strategy_used) or None if all fail.
        """
        use_auto = (len(self.strategy) == 1 and self.strategy[0] == "auto")
        ordered: List[InjectionType] = (
            ["cross_entity", "temporal", "numerical", "lexical"]
            if use_auto
            else self.strategy  # type: ignore[assignment]
        )

        for strat in ordered:
            result = None
            if strat == "cross_entity":
                result = self._cross_entity(triple, pred_index)
            elif strat == "temporal":
                result = self._temporal(triple)
            elif strat == "numerical":
                result = self._numerical(triple)
            elif strat == "lexical":
                result = self._lexical(triple)

            if result is not None:
                new_obj = result
                return (
                    Triple(
                        subject=triple.subject,
                        predicate=triple.predicate,
                        obj=new_obj,
                        source_doc=triple.source_doc,   # will be overwritten
                    ),
                    strat,
                )

        logger.debug(f"No strategy succeeded for triple: {triple}")
        return None

    # ------------------------------------------------------------------
    # Individual strategies
    # ------------------------------------------------------------------

    def _cross_entity(
        self, triple: Triple, pred_index: Dict[str, List[Triple]]
    ) -> Optional[str]:
        """
        Find another entity's object for the same predicate.
        Filters out triples from the same subject and identical objects.
        """
        pred_key = triple.predicate.strip().lower()
        candidates = [
            t for t in pred_index.get(pred_key, [])
            if t.subject.strip().lower() != triple.subject.strip().lower()
            and t.obj.strip().lower() != triple.obj.strip().lower()
        ]
        if not candidates:
            return None
        chosen = self._rng.choice(candidates)
        return chosen.obj

    def _temporal(self, triple: Triple) -> Optional[str]:
        """
        Find a 4-digit year in the object string and shift it by ±2–9 years.
        """
        match = re.search(r"\b(1[0-9]{3}|20[0-9]{2})\b", triple.obj)
        if not match:
            return None
        year = int(match.group())
        offset = self._rng.choice([-1, 1]) * self._rng.randint(2, 9)
        new_year = year + offset
        return triple.obj[:match.start()] + str(new_year) + triple.obj[match.end():]

    def _numerical(self, triple: Triple) -> Optional[str]:
        """
        Find any integer or decimal number and perturb it by ±5–25%.
        """
        match = re.search(r"\b\d+(\.\d+)?\b", triple.obj)
        if not match:
            return None
        original_num = float(match.group())
        if original_num == 0:
            return None
        pct = self._rng.uniform(0.05, 0.25) * self._rng.choice([-1, 1])
        new_num = original_num * (1 + pct)
        # Preserve int vs float formatting
        if "." in match.group():
            new_str = f"{new_num:.2f}"
        else:
            new_str = str(int(round(new_num)))
        return triple.obj[:match.start()] + new_str + triple.obj[match.end():]

    def _lexical(self, triple: Triple) -> Optional[str]:
        """
        Replace the object with a near-opposite from the lookup table.
        Matching is case-insensitive.
        """
        obj_lower = triple.obj.strip().lower()
        if obj_lower in _LEXICAL_TABLE:
            replacements = _LEXICAL_TABLE[obj_lower]
            return self._rng.choice(replacements)
        # Partial match: check if any key is a substring
        for key, replacements in _LEXICAL_TABLE.items():
            if key in obj_lower:
                return self._rng.choice(replacements)
        return None


# ---------------------------------------------------------------------------
# Evaluation helper
# ---------------------------------------------------------------------------

def match_detected_to_ground_truth(
    ground_truth: List[InjectedConflict],
    detected_edges,          # List[ConflictEdge] from ConflictDetector
    subject_match: str = "exact",   # "exact" or "normalized"
) -> Dict:
    """
    Compare detected ConflictEdge objects against ground-truth InjectedConflict records.

    A detected edge is a True Positive if there exists a ground-truth record with
    the same (subject, predicate) pair (one side of the conflict must match).

    Parameters
    ----------
    ground_truth : List[InjectedConflict]
        Output of ConflictInjector.inject().ground_truth
    detected_edges : List[ConflictEdge]
        Output of ConflictDetector.detect()
    subject_match : str
        "exact"      — exact string match on subject (case-insensitive)
        "normalized" — lowercase + strip whitespace comparison

    Returns
    -------
    dict with keys: tp, fp, fn, precision, recall, f1
    """
    def normalize(s: str) -> str:
        return s.strip().lower()

    # Build ground-truth lookup: (subject_norm, predicate_norm) → InjectedConflict
    gt_lookup: Dict[Tuple[str, str], InjectedConflict] = {}
    for gt in ground_truth:
        key = (normalize(gt.subject), normalize(gt.predicate))
        gt_lookup[key] = gt

    matched_gt_keys = set()
    tp = 0
    fp = 0

    for edge in detected_edges:
        # A detected edge involves two subjects (triple_a.subject, triple_b.subject)
        # and a shared predicate.  Check if either side matches a GT record.
        subj_a = normalize(edge.triple_a.subject)
        subj_b = normalize(edge.triple_b.subject)
        pred   = normalize(edge.conflicting_predicate.split("/")[0])  # handle "p1 / p2"

        hit = False
        for subj in (subj_a, subj_b):
            key = (subj, pred)
            if key in gt_lookup and key not in matched_gt_keys:
                matched_gt_keys.add(key)
                hit = True
                break

        if hit:
            tp += 1
        else:
            fp += 1

    fn = len(ground_truth) - tp

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)

    return {
        "tp": tp, "fp": fp, "fn": fn,
        "precision": round(precision, 4),
        "recall":    round(recall,    4),
        "f1":        round(f1,        4),
        "n_ground_truth": len(ground_truth),
        "n_detected":     len(detected_edges),
    }
