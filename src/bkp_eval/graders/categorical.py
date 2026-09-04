"""Graders for the remaining answer_types: enum, string_normalized, month_set, clarification
(grader ids enum_v1, string_normalized_v1, month_set_v1, clarification_v1).
"""
from __future__ import annotations

import re

from bharat_units.errors import ParseError
from bharat_units.numerals import from_words

from bkp_eval.graders._common import GradeResult, has_hedge_language, is_refusal, normalize_loose, try_parse_json_object
from bkp_eval.graders.dates import _MONTHS

_BOOL_SYNONYMS = {
    "true": {"true", "yes", "correct"},
    "false": {"false", "no", "incorrect"},
}


def _extract_scalar(raw_response: str, key: str = "value") -> str | None:
    obj = try_parse_json_object(raw_response)
    if obj is not None and key in obj:
        return str(obj[key])
    return raw_response


def _from_words_lenient(text: str) -> int | None:
    """from_words(), but tolerant of leading prose ("The answer is forty-five thousand...").
    Tries the whole (punctuation-trimmed) string first, then drops one leading token at a time
    until a suffix parses cleanly. Trailing prose after the number is not handled - see
    string_normalized's module-level TODO scope note.
    """
    cleaned = text.strip().rstrip(".!?")
    tokens = cleaned.split()
    for i in range(len(tokens)):
        try:
            return from_words(" ".join(tokens[i:]))
        except ParseError:
            continue
    return None


def grade_enum(item: dict, raw_response: str, regime: str = "R2") -> GradeResult:
    gold = normalize_loose(str(item["gold"]["value"]))
    text = normalize_loose(_extract_scalar(raw_response) or "")

    if not text:
        return GradeResult(correct=False, reason="empty response", is_refusal=True)

    if gold in _BOOL_SYNONYMS:
        synonyms = _BOOL_SYNONYMS[gold]
        # match a whole token, not a substring (so "incorrect" doesn't satisfy gold "correct")
        tokens = set(re.findall(r"[a-z]+", text))
        correct = bool(tokens & synonyms)
    else:
        correct = gold == text or gold in text.split()

    if not correct and is_refusal(raw_response):
        return GradeResult(correct=False, reason="refusal or empty response", is_refusal=True)

    return GradeResult(
        correct=correct, parsed=text,
        reason="matches gold enum" if correct else f"parsed {text!r} != gold {gold!r}",
    )


def grade_string_normalized(item: dict, raw_response: str, regime: str = "R2") -> GradeResult:
    gold_value = str(item["gold"]["value"])
    mode = item["gold"].get("normalization")
    aliases = item.get("accepted_aliases") or []
    raw_text = _extract_scalar(raw_response) or ""

    if is_refusal(raw_text):
        return GradeResult(correct=False, reason="refusal or empty response", is_refusal=True)

    if mode == "indian_digit_grouping":
        # Exact-string grading (schema.json changelog 0.1.2): strip currency symbol/whitespace
        # only - digit grouping itself must be exactly right, nothing else is forgiven.
        strip_re = re.compile(r"[₹\s]|rs\.?|inr", re.IGNORECASE)
        parsed = strip_re.sub("", raw_text)
        gold_clean = strip_re.sub("", gold_value)
        correct = parsed == gold_clean
        return GradeResult(correct=correct, parsed=parsed, reason="exact digit-grouping match" if correct else f"{parsed!r} != {gold_clean!r}")

    if mode == "indian_word_form":
        # Prefer semantic comparison (parse both sides back to an integer) over string matching -
        # far more robust to harmless phrasing differences than a literal/alias match would be.
        parsed_int = _from_words_lenient(raw_text)
        if parsed_int is not None:
            gold_int = from_words(gold_value)
            correct = parsed_int == gold_int
            return GradeResult(correct=correct, parsed=parsed_int, reason="word-form value matches" if correct else f"{parsed_int} != {gold_int}")
        # fall through to the lenient string match below

    candidates = {normalize_loose(gold_value), *(normalize_loose(a) for a in aliases)}
    parsed = normalize_loose(raw_text)
    if mode == "unit_string":
        candidates = {c.rstrip("s") for c in candidates}
        parsed = parsed.rstrip("s")
    correct = parsed in candidates
    return GradeResult(
        correct=correct, parsed=parsed,
        reason="matches gold/alias" if correct else f"parsed {parsed!r} not in {sorted(candidates)}",
    )


def grade_month_set(item: dict, raw_response: str, regime: str = "R2") -> GradeResult:
    gold = set(item["gold"]["months"])

    obj = try_parse_json_object(raw_response)
    if obj is not None and "months" in obj:
        try:
            parsed = {int(m) for m in obj["months"]}
        except (TypeError, ValueError):
            parsed = set()
    else:
        text = raw_response.lower()
        parsed = {v for k, v in _MONTHS.items() if re.search(rf"\b{k}\b", text)}

    if not parsed:
        if is_refusal(raw_response):
            return GradeResult(correct=False, reason="refusal or empty response", is_refusal=True)
        return GradeResult(correct=None, needs_judge=True, reason="no months could be extracted")

    correct = parsed == gold
    return GradeResult(
        correct=correct, parsed=sorted(parsed),
        reason="months match" if correct else f"parsed {sorted(parsed)} != gold {sorted(gold)}",
    )


def grade_clarification(item: dict, raw_response: str, regime: str = "R2") -> GradeResult:
    """Correct here means "hedges / names the competing conventions" - the Ambiguity Handling
    metric. Its mirror image, Overconfidence, is a confident single scalar with no
    hedge; that's flagged in `extra["single_confident_value"]` for score.py to aggregate,
    rather than recomputed there from scratch.
    """
    obj = try_parse_json_object(raw_response)
    if obj is not None and "clarification_required" in obj:
        correct = bool(obj["clarification_required"])
        return GradeResult(
            correct=correct, parsed=obj, reason="explicit clarification_required flag",
            extra={"single_confident_value": not correct},
        )

    if is_refusal(raw_response):
        return GradeResult(correct=False, reason="refusal, not a clarification", is_refusal=True)

    hedges = has_hedge_language(raw_response)
    assumptions = item["gold"].get("acceptable_named_assumptions", [])
    mentioned = sum(
        1 for a in assumptions
        if (key := re.split(r"[:(]", a)[0].strip().lower()) and key in raw_response.lower()
    )
    correct = hedges or mentioned >= 2
    single_confident_value = not correct and bool(re.search(r"\d", raw_response))

    return GradeResult(
        correct=correct,
        reason="hedges or names >=2 conventions" if correct else "gives a single confident answer",
        extra={"single_confident_value": single_confident_value, "mentioned_assumptions": mentioned},
    )
