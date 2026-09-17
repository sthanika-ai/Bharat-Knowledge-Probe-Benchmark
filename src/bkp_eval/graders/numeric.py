"""Graders for answer_type numeric / numeric_with_unit (grader ids numeric_v1 / numeric_unit_v1).

Also owns OOM ("order of magnitude") error detection - the "lakh/crore catastrophe rate" metric,
a >=10x-off answer that a lakh/crore-vs-million/billion mixup produces - since it only makes
sense in the same place the numeric comparison already happens.

Extraction scope (v1, R2 free-text path): a bare number (with commas, a currency prefix, or a
single trailing Indian/international scale word - "1.2 lakh crore", "₹83,000", "10 million") via
bharat_units.numerals, same as the item corpus itself was authored against. This does not attempt
full arithmetic-expression NLU ("half of 10 lakh minus 3 thousand") - a response phrased that way
will fail to extract and gets recorded as ungraded/needs_judge rather than a silent wrong mark.
"""
from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation

from bharat_units.errors import BharatUnitsError
from bharat_units.numerals import parse_indian_number

from bkp_eval.graders._common import GradeResult, is_refusal, try_parse_json_object

# Known-bias fix: has_explicit_unit used to check ONLY the Latin-script gold_unit string against
# the raw response, so a correct answer stated entirely in Devanagari (or another Indic script)
# was scored as missing its unit - see the Unit Discipline / accuracy caveat this was inflating.
# item["accepted_aliases"] (a generic schema field, already used by grade_string_normalized's
# unit_string mode for the same purpose) is now checked too. These are curated, high-confidence
# Devanagari renderings for the units actually used in numeric_with_unit items in this corpus -
# common international/scientific units and native Hindi numeral words with one standard,
# uncontroversial spelling (lakh/crore/arab/kharab/neel/padma). Deliberately NOT covering the
# regional/dialectal traditional land units (katha, biswa, marla, sarsai, kani, kranta, ...) -
# their Devanagari orthography varies by state/dialect and hand-typing one "canonical" spelling
# without a source would repeat the exact kind of unsourced-content problem this benchmark exists
# to avoid. That gap is real and stated in the report's Limitations, not silently closed here.
DEVANAGARI_UNIT_ALIASES: dict[str, tuple[str, ...]] = {
    "kilogram": ("किलोग्राम", "किलो"),
    "kilograms": ("किलोग्राम", "किलो"),
    "gram": ("ग्राम",),
    "grams": ("ग्राम",),
    "tonne": ("टन",),
    "tonnes": ("टन",),
    "quintal": ("क्विंटल",),
    "quintals": ("क्विंटल",),
    "lakh": ("लाख",),
    "crore": ("करोड़",),
    "arab": ("अरब",),
    "kharab": ("खरब",),
    "neel": ("नील",),
    "padma": ("पद्म",),
    "million": ("मिलियन",),
    "billion": ("बिलियन",),
    "trillion": ("ट्रिलियन",),
    "acre": ("एकड़",),
    "hectare": ("हेक्टेयर",),
    "sq_ft": ("वर्ग फुट", "वर्गफुट"),
    "sq_m": ("वर्ग मीटर", "वर्गमीटर"),
    "sq_yd": ("वर्ग गज", "वर्गगज"),
    "ft": ("फुट",),
    "rupees": ("रुपये", "रुपए"),
    "dollars": ("डॉलर",),
    "us dollars": ("अमेरिकी डॉलर", "डॉलर"),
    "percent": ("प्रतिशत",),
}


def _unit_candidates(item: dict, gold_unit: str) -> list[str]:
    """gold_unit plus its curated Devanagari aliases (if any are known for it) plus whatever
    per-item accepted_aliases the corpus itself already carries - see DEVANAGARI_UNIT_ALIASES's
    docstring-comment for what's covered and what's deliberately not.
    """
    candidates = [gold_unit, *DEVANAGARI_UNIT_ALIASES.get(gold_unit.lower().strip(), ())]
    candidates.extend(item.get("accepted_aliases") or [])
    return candidates


# Second known bug, found auditing the real "hardest items" (all 15 at 0% mean accuracy across
# the full 21-model roster): 8 of them were every "MSP for <crop> is Rs.X per quintal - express in
# Rs. per kg" item in C2, and ALL scored 0% because gold.unit spells out "rupees per kg" while
# every BKP-500 prompt itself states amounts using the abbreviation ("Rs.2275"), never the word
# "rupees" - so models overwhelmingly answered "Rs. 22.75 per kg" / "Rs/kg" / "Rs./kg" (objectively
# correct) and got marked wrong for not saying "rupees". A flat substring check on the whole
# spelled-out compound phrase also broke on "Rs/kg"-style responses, which don't contain the
# literal word "per" at all. This affects 17 numeric_with_unit items corpus-wide (10 in C2, 7 in
# C6) - every one with "rupee(s)" or "dollar(s)" anywhere in its gold unit.
_CURRENCY_INDICATORS: dict[str, tuple[str, ...]] = {
    "rupee": ("rupee", "rupees", "rs.", "rs", "₹", "inr"),
    "dollar": ("dollar", "dollars", "$", "usd"),
}


def _has_unit(candidate_unit: str, parsed_unit_lower: str) -> bool:
    """Is `candidate_unit` present in the (already-lowercased) response text? Checked word-by-word
    rather than as one literal phrase, for a compound unit like "rupees per kg" - a currency word
    is matched against its abbreviation/symbol set (_CURRENCY_INDICATORS) instead of requiring the
    spelled-out word, and the literal word "per" is never required (real responses commonly use
    "/" instead: "Rs/kg", "Rs./kg"). Every other word (kg, hectare, year, family, ...) still needs
    to appear, same as the original plain-substring check for a single-word unit like "grams".
    """
    normalized = candidate_unit.lower().replace("us dollar", "dollar")
    for word in normalized.split():
        if word == "per":
            continue
        stem = word.rstrip("s")
        if stem in _CURRENCY_INDICATORS:
            if not any(ind in parsed_unit_lower for ind in _CURRENCY_INDICATORS[stem]):
                return False
        elif stem not in parsed_unit_lower:
            return False
    return True

_SCALE_WORD_RE = (
    r"(?:hundreds?|thousands?|k|lakhs?|lacs?|crores?|cr|arabs?|kharabs?|neel|nils?|"
    r"padmas?|shankhs?|millions?|mn|billions?|bn|trillions?|tn|"
    r"quadrillions?|quintillions?|sextillions?)"
)
# <number>[<space><scale word>[<space><scale word>]]  e.g. "1.2 lakh crore", "83,000", "10 million"
_NUMBER_SPAN_RE = re.compile(
    rf"[₹$]?\s*-?\d[\d,]*(?:\.\d+)?(?:\s*{_SCALE_WORD_RE}(?:\s*{_SCALE_WORD_RE})?)?",
    re.IGNORECASE,
)


def _extract_number(text: str) -> Decimal | None:
    for match in _NUMBER_SPAN_RE.finditer(text):
        span = match.group(0).strip()
        if not span or not re.search(r"\d", span):
            continue
        try:
            return parse_indian_number(span)
        except BharatUnitsError:
            continue
    return None


def _within_tolerance(parsed: Decimal, gold: Decimal, tolerance: dict | None) -> bool:
    if tolerance is None or tolerance.get("type") == "exact":
        return parsed == gold
    value = Decimal(str(tolerance["value"]))
    if tolerance["type"] == "absolute":
        return abs(parsed - gold) <= value
    if tolerance["type"] == "relative":
        if gold == 0:
            return parsed == 0
        return abs(parsed - gold) <= value * abs(gold)
    raise ValueError(f"unknown tolerance type {tolerance['type']!r}")


_BARE_NUMBER_RE = re.compile(r"[₹$]?\s*-?\d[\d,]*(?:\.\d+)?")


def _extract_value_for_unit(text: str, unit_word: str) -> Decimal | None:
    """Extract the numeric coefficient that goes with `unit_word` - deliberately NOT
    `_extract_number`'s scale-multiplying parse, because numeric_with_unit stores the value and
    unit as separate fields (gold {"value": 10, "unit": "million"} means "10 million", not
    "10" pre-multiplied by a million). A bare-number regex plus "prefer the one right next to the
    unit word" handles a response that echoes the question's own number before stating the
    answer (e.g. "1 crore is 10 million." must pick 10, not 1).
    """
    candidates = list(_BARE_NUMBER_RE.finditer(text))
    if not candidates:
        return None
    unit_stem = re.escape(unit_word.lower().rstrip("s"))
    near_re = re.compile(rf"({_BARE_NUMBER_RE.pattern})\s*{unit_stem}", re.IGNORECASE)
    near_matches = list(near_re.finditer(text))
    span = near_matches[-1].group(1) if near_matches else candidates[-1].group(0)
    cleaned = re.sub(r"[₹$,]", "", span).strip()
    try:
        return Decimal(cleaned)
    except InvalidOperation:
        return None


def _oom_error(parsed: Decimal, gold: Decimal) -> bool | None:
    """True if `parsed` is off from `gold` by at least one order of magnitude (>=10x too big
    or too small) - the lakh/crore/million/billion confusion this benchmark exists to catch.
    None when the comparison is undefined (gold is exactly zero).
    """
    if gold == 0:
        return None
    if parsed == 0:
        return True
    ratio = abs(parsed / gold)
    return ratio >= 10 or ratio <= Decimal("0.1")


def grade_numeric(item: dict, raw_response: str, regime: str = "R2") -> GradeResult:
    gold = Decimal(str(item["gold"]["value"]))
    tolerance = item.get("tolerance")

    obj = try_parse_json_object(raw_response)
    if obj is not None and "value" in obj:
        try:
            parsed = Decimal(str(obj["value"]))
        except InvalidOperation:
            parsed = None
    else:
        parsed = _extract_number(raw_response)

    if parsed is None:
        if is_refusal(raw_response):
            return GradeResult(correct=False, reason="refusal or empty response", is_refusal=True)
        return GradeResult(correct=None, needs_judge=True, reason="no number could be extracted")

    correct = _within_tolerance(parsed, gold, tolerance)
    extra = {} if correct else {"oom_error": _oom_error(parsed, gold)}
    return GradeResult(
        correct=correct,
        parsed=parsed,
        reason="within tolerance" if correct else f"parsed {parsed} != gold {gold}",
        extra=extra,
    )


def grade_numeric_with_unit(item: dict, raw_response: str, regime: str = "R2") -> GradeResult:
    gold_value = Decimal(str(item["gold"]["value"]))
    gold_unit = str(item["gold"]["unit"])
    tolerance = item.get("tolerance")

    obj = try_parse_json_object(raw_response)
    if obj is not None and "value" in obj:
        try:
            parsed_value: Decimal | None = Decimal(str(obj["value"]))
        except InvalidOperation:
            parsed_value = None
        parsed_unit = str(obj.get("unit", ""))
    else:
        parsed_value = _extract_value_for_unit(raw_response, gold_unit)
        parsed_unit = raw_response

    if parsed_value is None:
        if is_refusal(raw_response):
            return GradeResult(correct=False, reason="refusal or empty response", is_refusal=True)
        return GradeResult(correct=None, needs_judge=True, reason="no number could be extracted")

    value_correct = _within_tolerance(parsed_value, gold_value, tolerance)
    # Unit Discipline: does the response carry the gold unit at all, explicitly -
    # tracked independently of whether the *value* is right, so a right-number-wrong-unit answer
    # (or a bare number with the unit silently dropped) shows up in that metric either way. Checks
    # gold_unit's curated Devanagari aliases and currency abbreviations too - see
    # DEVANAGARI_UNIT_ALIASES and _has_unit above for why these exist and what they do/don't cover.
    parsed_unit_lower = parsed_unit.lower()
    has_explicit_unit = any(_has_unit(c, parsed_unit_lower) for c in _unit_candidates(item, gold_unit))
    correct = value_correct and has_explicit_unit

    extra = {"has_explicit_unit": has_explicit_unit}
    if not value_correct:
        extra["oom_error"] = _oom_error(parsed_value, gold_value)

    return GradeResult(
        correct=correct,
        parsed={"value": parsed_value, "unit": parsed_unit},
        reason=(
            "within tolerance and unit present" if correct
            else f"parsed ({parsed_value}, unit-present={has_explicit_unit}) "
                 f"!= gold ({gold_value}, {gold_unit!r})"
        ),
        extra=extra,
    )
