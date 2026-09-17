"""Shared plumbing used by every grader module: the result type, the R1/R2 response-unwrapping
helper, and text-normalization utilities that more than one grader needs.

Design note on R1 vs R2: R1 responses are a JSON object (e.g. {"value": ...,
"unit": ...}); grading R1 is just a dict lookup. R2 responses are free text; grading R2 means
extracting a candidate value with the same heuristics a careful human skimming the reply would
use, then comparing that extraction to gold exactly like R1 would. Every grader below tries the
R1 (JSON) path first regardless of the declared regime - a model asked for free text sometimes
replies with JSON anyway, and there's no reason to penalize that.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

REFUSAL_PHRASES = (
    "i cannot", "i can't", "i don't know", "i do not know", "as an ai",
    "i'm not able to", "i am not able to", "unable to determine", "no information",
    "i don't have", "i do not have",
)

HEDGE_PHRASES = (
    "depends", "ambiguous", "which convention", "could mean", "two values",
    "two different", "clarify", "clarification", "not specified", "either",
    "varies by", "varies depending",
)


@dataclass
class GradeResult:
    """Outcome of grading one (item, response) pair.

    `correct` is None (not True/False) when the deterministic grader declines to rule and the
    cascade fell through to the judge stub without an actual judge configured - see
    graders/__init__.py:grade(). Never treat `correct is None` as either right or wrong when
    aggregating; score.py filters those into a separate "undetermined" bucket.
    """

    correct: bool | None
    parsed: Any = None
    reason: str = ""
    needs_judge: bool = False
    is_refusal: bool = False
    extra: dict = field(default_factory=dict)


def is_refusal(raw_response: str) -> bool:
    text = raw_response.strip().lower()
    if not text:
        return True
    return any(phrase in text for phrase in REFUSAL_PHRASES)


def has_hedge_language(raw_response: str) -> bool:
    text = raw_response.strip().lower()
    return any(phrase in text for phrase in HEDGE_PHRASES)


def try_parse_json_object(raw_response: str) -> dict | None:
    """Best-effort extraction of a JSON object from `raw_response`.

    Tries the whole string first (the R1-compliant case), then falls back to the first
    balanced-looking {...} substring, since models under R2 sometimes wrap JSON in prose or a
    markdown code fence ("Sure, here you go:\\n```json\\n{...}\\n```").
    """
    text = raw_response.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text).strip()
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except (json.JSONDecodeError, ValueError):
        pass

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            obj = json.loads(match.group(0))
            return obj if isinstance(obj, dict) else None
        except (json.JSONDecodeError, ValueError):
            return None
    return None


def normalize_loose(s: str) -> str:
    """Lowercase, strip, collapse whitespace, drop surrounding punctuation - the baseline
    normalization every grader applies before a string comparison, on top of whatever
    type-specific normalization (see string_normalized's `gold.normalization`) it also needs.
    """
    s = s.strip().lower()
    s = re.sub(r"\s+", " ", s)
    s = s.strip(" .!\"'")
    return s
