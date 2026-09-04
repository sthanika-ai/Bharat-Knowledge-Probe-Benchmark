"""Graders for answer_type date / date_range (grader ids date_v1 / date_range_v1).

Extraction scope (v1, R2 free-text path): ISO (2025-05-28), "28 May 2025" / "May 28, 2025", and
"28-05-2025" (day-month-year only - this benchmark's domain is India, and unqualified numeric
dates are day-first there; a model that means month-first and doesn't say so is exactly the kind
of locale-convention slip this benchmark is measuring, not something to guess around).
"""
from __future__ import annotations

import re
from datetime import date

from bkp_eval.graders._common import GradeResult, is_refusal, try_parse_json_object

_MONTHS = {
    "jan": 1, "january": 1, "feb": 2, "february": 2, "mar": 3, "march": 3,
    "apr": 4, "april": 4, "may": 5, "jun": 6, "june": 6, "jul": 7, "july": 7,
    "aug": 8, "august": 8, "sep": 9, "sept": 9, "september": 9, "oct": 10, "october": 10,
    "nov": 11, "november": 11, "dec": 12, "december": 12,
}
_MONTH_NAME_RE = "|".join(_MONTHS)

_ISO_RE = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")
_DMY_NUMERIC_RE = re.compile(r"\b(\d{1,2})[/-](\d{1,2})[/-](\d{4})\b")
_DAY_MONTH_YEAR_RE = re.compile(
    rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+({_MONTH_NAME_RE})\.?,?\s+(\d{{4}})\b", re.IGNORECASE
)
_MONTH_DAY_YEAR_RE = re.compile(
    rf"\b({_MONTH_NAME_RE})\.?\s+(\d{{1,2}})(?:st|nd|rd|th)?,?\s+(\d{{4}})\b", re.IGNORECASE
)


def _dates_in(text: str) -> list[date]:
    found: list[tuple[int, date]] = []
    for m in _ISO_RE.finditer(text):
        y, mo, d = (int(g) for g in m.groups())
        try:
            found.append((m.start(), date(y, mo, d)))
        except ValueError:
            pass
    for m in _DAY_MONTH_YEAR_RE.finditer(text):
        d, mo_name, y = m.groups()
        try:
            found.append((m.start(), date(int(y), _MONTHS[mo_name.lower()], int(d))))
        except ValueError:
            pass
    for m in _MONTH_DAY_YEAR_RE.finditer(text):
        mo_name, d, y = m.groups()
        try:
            found.append((m.start(), date(int(y), _MONTHS[mo_name.lower()], int(d))))
        except ValueError:
            pass
    for m in _DMY_NUMERIC_RE.finditer(text):
        d, mo, y = (int(g) for g in m.groups())
        try:
            found.append((m.start(), date(y, mo, d)))
        except ValueError:
            pass
    found.sort(key=lambda pair: pair[0])
    return [d for _, d in found]


def _parse_iso(s: str) -> date | None:
    try:
        return date.fromisoformat(s.strip())
    except ValueError:
        return None


def grade_date(item: dict, raw_response: str, regime: str = "R2") -> GradeResult:
    gold = _parse_iso(item["gold"]["value"])
    if gold is None:
        # A real invariant violation (corrupted corpus data), not a normal grading outcome -
        # raise rather than assert so it can't be silently stripped under `python -O`.
        raise ValueError(f"malformed gold date on {item['id']!r}")

    obj = try_parse_json_object(raw_response)
    if obj is not None and "value" in obj:
        parsed = _parse_iso(str(obj["value"]))
    else:
        candidates = _dates_in(raw_response)
        parsed = candidates[0] if candidates else None

    if parsed is None:
        if is_refusal(raw_response):
            return GradeResult(correct=False, reason="refusal or empty response", is_refusal=True)
        return GradeResult(correct=None, needs_judge=True, reason="no date could be extracted")

    correct = parsed == gold
    return GradeResult(
        correct=correct, parsed=parsed,
        reason="date matches" if correct else f"parsed {parsed} != gold {gold}",
    )


def grade_date_range(item: dict, raw_response: str, regime: str = "R2") -> GradeResult:
    gold_start = _parse_iso(item["gold"]["start"])
    gold_end = _parse_iso(item["gold"]["end"])
    if gold_start is None or gold_end is None:
        raise ValueError(f"malformed gold range on {item['id']!r}")

    obj = try_parse_json_object(raw_response)
    if obj is not None and "start" in obj and "end" in obj:
        parsed_start = _parse_iso(str(obj["start"]))
        parsed_end = _parse_iso(str(obj["end"]))
    else:
        candidates = _dates_in(raw_response)
        parsed_start = candidates[0] if len(candidates) >= 1 else None
        parsed_end = candidates[1] if len(candidates) >= 2 else None

    if parsed_start is None or parsed_end is None:
        if is_refusal(raw_response):
            return GradeResult(correct=False, reason="refusal or empty response", is_refusal=True)
        return GradeResult(
            correct=None, needs_judge=True,
            reason="fewer than two dates could be extracted",
        )

    correct = parsed_start == gold_start and parsed_end == gold_end
    return GradeResult(
        correct=correct,
        parsed={"start": parsed_start, "end": parsed_end},
        reason=(
            "range matches" if correct
            else f"parsed ({parsed_start}..{parsed_end}) != gold ({gold_start}..{gold_end})"
        ),
    )
