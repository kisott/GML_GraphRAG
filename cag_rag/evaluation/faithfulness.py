"""
Faithfulness Scorer  —  LLM-as-Judge
--------------------------------------
Evaluates generated answers on two dimensions:

  factual_score        (0-10)
      Does the answer stay grounded in the retrieved context?
      Penalises hallucination and statements unsupported by context.

  conflict_score       (0-10)
      If the context contains conflicting information, does the answer
      surface the disagreement explicitly rather than silently picking one side?
      If NO conflicts are present in context, full marks (10) are given
      automatically — the system cannot be penalised for a non-event.

  overall              average of the two scores, rescaled to [0, 1]

Usage
-----
    from cag_rag.evaluation.faithfulness import FaithfulnessScorer

    scorer = FaithfulnessScorer(api_key=os.environ["OPENAI_API_KEY"])
    result = scorer.score(
        question="Who won Super Bowl 50?",
        answer="Denver Broncos won Super Bowl 50.",
        context="Denver Broncos beat the Carolina Panthers 24-10.",
        has_conflicts=False,
    )
    print(result.factual_score, result.conflict_score, result.overall)
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import dataclass
from typing import List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Judge prompt
# ---------------------------------------------------------------------------

_JUDGE_SYSTEM = """You are an impartial judge evaluating a question-answering system.
Your task is to score a generated answer on two dimensions.
Always respond with a single valid JSON object — no markdown, no extra text."""

_JUDGE_USER_TEMPLATE = """## Question
{question}

## Retrieved Context
{context}

## Generated Answer
{answer}

## Scoring Instructions

Score the answer on TWO dimensions (integer 0–10 each):

1. factual_score — Factual Faithfulness
   Does the answer stay grounded in the retrieved context?
   10 = answer uses only information present in context, accurately
    5 = mostly faithful but includes minor unsupported claims
    0 = ignores context or directly contradicts it

2. conflict_score — Conflict Acknowledgment
   Context contains conflicts: {has_conflicts}
   - If False (no conflicts in context): set conflict_score = 10 automatically.
   - If True (conflicts present in context):
     10 = answer explicitly acknowledges ALL conflicts and presents both sides
      5 = partially acknowledges conflict but partially resolves it silently
      0 = silently picks one side without acknowledging the disagreement

Return ONLY this JSON (no markdown fences):
{{"factual_score": <int>, "conflict_score": <int>, "reasoning": "<one sentence>"}}"""


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class FaithfulnessResult:
    question: str
    factual_score: int       # 0-10
    conflict_score: int      # 0-10
    overall: float           # average rescaled to [0, 1]
    reasoning: str
    raw_response: str = ""

    def __str__(self) -> str:
        return (
            f"Factual={self.factual_score}/10  "
            f"Conflict={self.conflict_score}/10  "
            f"Overall={self.overall:.2f}  |  {self.reasoning}"
        )


# ---------------------------------------------------------------------------
# FaithfulnessScorer
# ---------------------------------------------------------------------------

class FaithfulnessScorer:
    """
    LLM-as-judge faithfulness scorer for RAG answer quality.

    Parameters
    ----------
    api_key : str
        OpenAI API key.
    model_name : str
        OpenAI model to use as the judge. gpt-4o-mini is fast and cheap.
    retries : int
        Number of retries on API errors.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "gpt-4o-mini",
        retries: int = 3,
    ):
        self.model_name = model_name
        self.retries = retries

        key = api_key or os.environ.get("OPENAI_API_KEY", "")
        if not key:
            raise ValueError("OPENAI_API_KEY not set.")
        try:
            from openai import OpenAI
            self._client = OpenAI(api_key=key)
        except ImportError:
            raise ImportError("pip install openai")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def score(
        self,
        question: str,
        answer: str,
        context: str,
        has_conflicts: bool,
    ) -> FaithfulnessResult:
        """
        Score one answer.

        Parameters
        ----------
        question : str
        answer : str
            The generated answer to evaluate.
        context : str
            The retrieved context shown to the system when generating the answer.
            For Standard RAG: the raw passages.
            For graph systems: the triple text.
            For CAG-RAG: triples + conflict blocks.
        has_conflicts : bool
            Whether the retrieved context contains at least one conflict block.
        """
        user_prompt = _JUDGE_USER_TEMPLATE.format(
            question=question,
            context=context[:3000],   # cap to avoid very long prompts
            answer=answer,
            has_conflicts=has_conflicts,
        )

        raw = self._call_with_retry(_JUDGE_SYSTEM, user_prompt)
        return self._parse(question, raw)

    def score_batch(
        self,
        questions: List[str],
        answers: List[str],
        contexts: List[str],
        has_conflicts_list: List[bool],
    ) -> List[FaithfulnessResult]:
        """Score multiple answers sequentially."""
        return [
            self.score(q, a, c, hc)
            for q, a, c, hc in zip(questions, answers, contexts, has_conflicts_list)
        ]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _call_with_retry(self, system_prompt: str, user_prompt: str) -> str:
        last_exc = None
        for attempt in range(self.retries):
            try:
                response = self._client.chat.completions.create(
                    model=self.model_name,
                    temperature=0.0,
                    max_tokens=256,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user",   "content": user_prompt},
                    ],
                )
                return response.choices[0].message.content.strip()
            except Exception as exc:
                last_exc = exc
                wait = 2 ** attempt
                logger.warning(f"Judge API error (attempt {attempt+1}): {exc}. Retrying in {wait}s …")
                time.sleep(wait)
        raise RuntimeError("Judge API failed") from last_exc

    def _parse(self, question: str, raw: str) -> FaithfulnessResult:
        """Parse JSON response from the judge; fall back to neutral scores on failure."""
        try:
            # Strip accidental markdown fences
            clean = re.sub(r"```[a-z]*\n?|```", "", raw).strip()
            data = json.loads(clean)
            factual  = max(0, min(10, int(data.get("factual_score",  5))))
            conflict = max(0, min(10, int(data.get("conflict_score", 5))))
            reasoning = str(data.get("reasoning", ""))
        except Exception as e:
            logger.warning(f"Failed to parse judge response: {e!r}\nRaw: {raw!r}")
            factual, conflict, reasoning = 5, 5, f"parse error: {raw[:80]}"

        overall = (factual + conflict) / 20.0   # rescale to [0, 1]
        return FaithfulnessResult(
            question=question,
            factual_score=factual,
            conflict_score=conflict,
            overall=overall,
            reasoning=reasoning,
            raw_response=raw,
        )
