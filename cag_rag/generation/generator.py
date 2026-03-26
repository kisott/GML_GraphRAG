"""
Generation Module
-----------------
Takes a RetrievalResult from the conflict-aware retriever and generates
a final answer using the OpenAI ChatGPT API.

Two prompt modes
----------------
standard       — facts only, no conflict context (mirrors Standard RAG behavior)
conflict_aware — facts + structured conflict blocks (CAG-RAG behavior)

Both modes use the same model so that generation quality differences
are attributable solely to the retrieved context, not the model.

Usage
-----
    import os
    from cag_rag.generation import OpenAIGenerator

    gen = OpenAIGenerator(api_key=os.environ["OPENAI_API_KEY"])
    result = gen.generate(query, retrieval_result, mode="conflict_aware")
    print(result.answer)
    print(result.flagged_conflict)   # True if LLM reported a conflict
"""

from __future__ import annotations

import logging
import os
import re
import time
from dataclasses import dataclass, field
from typing import Literal, Optional

from ..retrieval.conflict_aware import RetrievalResult

logger = logging.getLogger(__name__)

GenerationMode = Literal["standard", "conflict_aware"]

# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

_SYSTEM_STANDARD = """You are a factual question-answering assistant.
Answer the question using ONLY the provided facts.
If the facts are insufficient, say "I cannot determine the answer from the given information."
Be concise and direct."""

_SYSTEM_CONFLICT_AWARE = """You are a factual question-answering assistant that handles contradictory evidence.
Answer the question using the provided facts.

If CONFLICT blocks are present:
  1. Acknowledge the contradiction explicitly.
  2. Present both conflicting claims and their sources.
  3. Do NOT silently pick one side — report the disagreement.
  4. If one source is more reliable or recent, you may note that, but still surface both.

If no conflicts are present, answer directly and concisely."""


# ---------------------------------------------------------------------------
# PromptBuilder
# ---------------------------------------------------------------------------

class PromptBuilder:
    """
    Builds the user-turn prompt from a RetrievalResult.

    mode="standard"       → only the fact triples, no conflict blocks
    mode="conflict_aware" → fact triples + [CONFLICT] blocks
    """

    @staticmethod
    def build(
        query: str,
        retrieval_result: RetrievalResult,
        mode: GenerationMode = "conflict_aware",
    ) -> str:
        lines = []

        if retrieval_result.subgraph_triples:
            lines.append("## Retrieved Facts")
            for t in retrieval_result.subgraph_triples:
                lines.append(
                    f"- {t['subject']} —[{t['predicate']}]→ {t['object']} "
                    f"(source: {t.get('source_doc', 'unknown')})"
                )
        else:
            lines.append("## Retrieved Facts\n(none)")

        if mode == "conflict_aware" and retrieval_result.conflict_contexts:
            lines.append("\n## Detected Conflicts")
            for cc in retrieval_result.conflict_contexts:
                lines.append(cc.to_prompt_text())

        lines.append(f"\n## Question\n{query}")
        lines.append("\n## Answer")

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# GenerationResult
# ---------------------------------------------------------------------------

@dataclass
class GenerationResult:
    """Output of one generation call."""
    query: str
    mode: GenerationMode
    answer: str
    flagged_conflict: bool
    prompt_tokens: int = 0
    completion_tokens: int = 0
    model: str = ""
    raw_response: object = field(default=None, repr=False)

    def __str__(self) -> str:
        flag = " [CONFLICT FLAGGED]" if self.flagged_conflict else ""
        return f"[{self.mode}]{flag}\nQ: {self.query}\nA: {self.answer}"


# ---------------------------------------------------------------------------
# OpenAIGenerator
# ---------------------------------------------------------------------------

class OpenAIGenerator:
    """
    Wraps the OpenAI ChatCompletion API for CAG-RAG generation experiments.

    Parameters
    ----------
    api_key : str
        OpenAI API key. Defaults to os.environ["OPENAI_API_KEY"].
    model_name : str
        OpenAI model identifier. Default: "gpt-4o-mini".
    temperature : float
        Sampling temperature (0 = deterministic).
    max_tokens : int
        Maximum tokens in the generated answer.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "gpt-4o-mini",
        temperature: float = 0.0,
        max_tokens: int = 512,
    ):
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens

        key = api_key or os.environ.get("OPENAI_API_KEY", "")
        if not key:
            raise ValueError(
                "OpenAI API key required. Pass api_key= or set OPENAI_API_KEY env var."
            )

        try:
            from openai import OpenAI
            self._client = OpenAI(api_key=key)
        except ImportError:
            raise ImportError(
                "openai not installed. Run: pip install openai"
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self,
        query: str,
        retrieval_result: RetrievalResult,
        mode: GenerationMode = "conflict_aware",
        retries: int = 3,
    ) -> GenerationResult:
        """
        Generate an answer for query given the retrieval result.

        Parameters
        ----------
        query : str
        retrieval_result : RetrievalResult
        mode : "standard" | "conflict_aware"
        retries : int
        """
        system_prompt = (
            _SYSTEM_CONFLICT_AWARE if mode == "conflict_aware" else _SYSTEM_STANDARD
        )
        user_prompt = PromptBuilder.build(query, retrieval_result, mode=mode)

        logger.info(f"Generating [{mode}] for query: '{query}'")

        raw = self._call_with_retry(system_prompt, user_prompt, retries=retries)

        answer_text = raw.choices[0].message.content.strip()
        flagged = _detect_conflict_flag(answer_text)

        usage = raw.usage
        return GenerationResult(
            query=query,
            mode=mode,
            answer=answer_text,
            flagged_conflict=flagged,
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
            model=self.model_name,
            raw_response=raw,
        )

    def generate_both_modes(
        self,
        query: str,
        retrieval_result: RetrievalResult,
    ) -> dict:
        """
        Run both standard and conflict_aware modes for the same query.
        Returns {"standard": GenerationResult, "conflict_aware": GenerationResult}.
        """
        return {
            "standard":       self.generate(query, retrieval_result, mode="standard"),
            "conflict_aware": self.generate(query, retrieval_result, mode="conflict_aware"),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _call_with_retry(self, system_prompt: str, user_prompt: str, retries: int = 3):
        last_exc = None
        for attempt in range(retries):
            try:
                response = self._client.chat.completions.create(
                    model=self.model_name,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user",   "content": user_prompt},
                    ],
                )
                return response
            except Exception as exc:
                last_exc = exc
                wait = 2 ** attempt
                logger.warning(
                    f"OpenAI API error (attempt {attempt+1}/{retries}): {exc}. "
                    f"Retrying in {wait}s …"
                )
                time.sleep(wait)
        raise RuntimeError(f"OpenAI API failed after {retries} attempts") from last_exc


# ---------------------------------------------------------------------------
# Conflict flagging heuristic
# ---------------------------------------------------------------------------

_CONFLICT_PATTERNS = re.compile(
    r"\b(conflict|contradict|inconsisten|disagree|dispute|discrepan"
    r"|two sources|both sources|one source says|another source|however.*source"
    r"|source [ab] (says|claims|states)|unclear|ambiguous)\b",
    re.IGNORECASE,
)


def _detect_conflict_flag(text: str) -> bool:
    """
    Returns True if the generated answer explicitly acknowledges a conflict.
    Used to compute Conflict Coverage metric.
    """
    return bool(_CONFLICT_PATTERNS.search(text))
