"""Builds the two prompt regimes the harness evaluates every item under:

R1 strict - a system prompt demands a JSON object shaped like the item's own `gold` field.
Isolates knowledge from writing style; graders/_common.py's try_parse_json_object() is the
matching consumer on the other end.

R2 natural - the bare item prompt, free text, no formatting demand. Measures what a real user
gets; graders fall back to text-extraction heuristics for this regime.
"""
from __future__ import annotations

_JSON_SHAPE_HINT = {
    "numeric": '{"value": <number>}',
    "numeric_with_unit": '{"value": <number>, "unit": "<string>"}',
    "date": '{"value": "<YYYY-MM-DD>"}',
    "date_range": '{"start": "<YYYY-MM-DD>", "end": "<YYYY-MM-DD>"}',
    "enum": '{"value": "<string>"}',
    "string_normalized": '{"value": "<string>"}',
    "month_set": '{"months": [<int>, ...]}',
    "clarification": '{"clarification_required": <bool>, "reason": "<string>"}',
}

R1_SYSTEM_TEMPLATE = (
    "Answer with ONLY a single JSON object, no prose, no markdown fence, matching exactly this "
    "shape: {shape}. If the question is genuinely ambiguous without more information, respond "
    'with {{"clarification_required": true, "reason": "<why>"}} instead.'
)


def build_prompt(item: dict, regime: str) -> tuple[str | None, str]:
    """Returns (system, user_prompt) for `item` under `regime` ("R1" or "R2")."""
    if regime == "R1":
        shape = _JSON_SHAPE_HINT[item["answer_type"]]
        return R1_SYSTEM_TEMPLATE.format(shape=shape), item["prompt"]
    if regime == "R2":
        return None, item["prompt"]
    raise ValueError(f"unknown regime {regime!r} - expected 'R1' or 'R2'")
