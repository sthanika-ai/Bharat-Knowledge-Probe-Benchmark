"""Grader cascade tests.

`test_gold_answer_always_grades_correct` is the load-bearing one: it runs every one of the
~1084 real items+controls through grade() with its OWN gold answer wrapped as an R1 JSON
response, and asserts a correct verdict. If a grader is wrong in a way that would reject a
right answer, this fails against real corpus data, not a hand-picked fixture - same spirit as
tests/test_corpus_validation.py running validate_items.py over the real data.

Everything else here targets one behavior at a time with small, explicit fixtures.
"""
from __future__ import annotations

import json
import re

import pytest

from bkp_eval.graders import grade
from bkp_eval.items import load_corpus

CORPUS = load_corpus()
ALL_RECORDS = list(CORPUS.all_by_id.values())


def _gold_as_r1_response(item: dict) -> str:
    gold = item["gold"]
    if item["answer_type"] == "clarification":
        return json.dumps({"clarification_required": True})
    return json.dumps(gold)


@pytest.mark.parametrize("item", ALL_RECORDS, ids=lambda item: item["id"])
def test_gold_answer_always_grades_correct(item):
    response = _gold_as_r1_response(item)
    result = grade(item, response, regime="R1")
    assert result.correct is True, f"{item['id']}: gold answer graded {result.correct} - {result.reason}"


# ---------------------------------------------------------------------------
# numeric / numeric_with_unit
# ---------------------------------------------------------------------------

NUMERIC_ITEM = CORPUS.items["BKP-C1-0001"]  # "Express 5 lakh as a plain number" -> 500000
UNIT_ITEM = CORPUS.items["BKP-C1-0002"]  # "How many million is 1 crore?" -> 10.0 million


def test_numeric_plain_text_correct():
    assert grade(NUMERIC_ITEM, "500000").correct is True


def test_numeric_wrong_value():
    result = grade(NUMERIC_ITEM, "50000")
    assert result.correct is False


def test_numeric_oom_error_flagged_on_10x_slip():
    # 5,000,000 instead of 500,000 - exactly the lakh/crore-confusion failure mode this exists to catch.
    result = grade(NUMERIC_ITEM, "5000000")
    assert result.correct is False
    assert result.extra["oom_error"] is True


def test_numeric_near_miss_not_flagged_as_oom():
    result = grade(NUMERIC_ITEM, "500001")  # off by 1, outside exact tolerance but not an OOM error
    assert result.correct is False
    assert result.extra["oom_error"] is False


def test_numeric_refusal():
    result = grade(NUMERIC_ITEM, "I don't know.")
    assert result.correct is False
    assert result.is_refusal is True


def test_numeric_unparseable_declines_rather_than_guesses():
    result = grade(NUMERIC_ITEM, "somewhere around a lot")
    assert result.correct is None
    assert result.needs_judge is True


def test_numeric_with_unit_correct_value_and_unit():
    result = grade(UNIT_ITEM, "1 crore is 10 million.")
    assert result.correct is True
    assert result.extra["has_explicit_unit"] is True


def test_numeric_with_unit_correct_value_missing_unit_fails_but_flags_unit_discipline():
    result = grade(UNIT_ITEM, "10")
    assert result.correct is False
    assert result.extra["has_explicit_unit"] is False


def test_numeric_with_unit_r1_json():
    result = grade(UNIT_ITEM, json.dumps({"value": 10, "unit": "million"}), regime="R1")
    assert result.correct is True


def test_numeric_with_unit_accepts_curated_devanagari_alias():
    # the known-bias fix: a correct answer stated entirely in Devanagari must not be scored as
    # missing its unit just because "million" (Latin) never appears in the response.
    result = grade(UNIT_ITEM, "1 करोड़ = 10 मिलियन")
    assert result.correct is True
    assert result.extra["has_explicit_unit"] is True


def test_numeric_with_unit_devanagari_alias_also_works_in_r1_json():
    result = grade(UNIT_ITEM, json.dumps({"value": 10, "unit": "मिलियन"}), regime="R1")
    assert result.correct is True


def test_numeric_with_unit_still_rejects_wrong_unit_even_in_devanagari():
    # "अरब" (arab) is not an alias of "million" - a plausible-looking but wrong Devanagari unit
    # must not be accepted just because SOME Devanagari word is present.
    result = grade(UNIT_ITEM, "10 अरब")
    assert result.extra["has_explicit_unit"] is False


def test_numeric_with_unit_respects_item_level_accepted_aliases_too():
    # a per-item accepted_aliases entry (the same mechanism grade_string_normalized's
    # unit_string mode already uses) must also be honored, not just the curated built-in table.
    item = dict(UNIT_ITEM)
    item["accepted_aliases"] = ["mn"]
    result = grade(item, "1 crore is 10 mn.")
    assert result.correct is True
    assert result.extra["has_explicit_unit"] is True


# ---------------------------------------------------------------------------
# currency-abbreviation fix (found auditing the real 0%-across-every-model "hardest items" - see
# numeric.py's _has_unit docstring for the full story)
# ---------------------------------------------------------------------------

MSP_ITEM = CORPUS.items["BKP-C2-0004"]  # "MSP for wheat is Rs.2275/quintal -> Rs./kg" -> 22.75 rupees per kg


@pytest.mark.parametrize("response", [
    # these are the ACTUAL (verbatim) responses collected in this run's results/*/full.jsonl for
    # this item, from 5 different models, every one of them objectively correct and every one of
    # them scored has_explicit_unit=False before this fix
    '{"value": 22.75, "unit": "Rs./kg"}',
    '{"value": 22.75, "unit": "Rs per kg"}',
    '{"value": 22.75, "unit": "Rs/kg"}',
    "The MSP for wheat is **Rs. 22.75 per kg**.",
])
def test_numeric_with_unit_accepts_real_world_rupee_abbreviation_responses(response):
    result = grade(MSP_ITEM, response, regime="R1" if response.startswith("{") else "R2")
    assert result.correct is True, result.reason
    assert result.extra["has_explicit_unit"] is True


def test_numeric_with_unit_currency_fix_still_requires_the_non_currency_words():
    # "Rs. 22.75" alone (no "kg"/"per kg") must NOT get unit credit just because a rupee
    # indicator is present - only the currency word is exempted from literal matching.
    result = grade(MSP_ITEM, '{"value": 22.75, "unit": "Rs."}')
    assert result.extra["has_explicit_unit"] is False


def test_numeric_with_unit_currency_fix_still_requires_the_right_currency():
    # a dollar sign must not satisfy a rupee-denominated unit
    result = grade(MSP_ITEM, '{"value": 22.75, "unit": "$/kg"}')
    assert result.extra["has_explicit_unit"] is False


def test_numeric_with_unit_us_dollars_alias_normalizes_correctly():
    item = dict(UNIT_ITEM)
    item["gold"] = {"value": 10.0, "unit": "US dollars"}
    result = grade(item, '{"value": 10, "unit": "USD"}')
    assert result.extra["has_explicit_unit"] is True


# ---------------------------------------------------------------------------
# quadrillion/quintillion/sextillion scale-word gap (found auditing C1 control items past
# trillion): _SCALE_WORD_RE only recognized scale words up to trillion, so a free-text response
# using one of these three words had its scale word silently dropped by the number-span regex -
# "1 quadrillion" extracted as the bare coefficient 1, not 1e15 - rather than failing to parse
# (which would at least have surfaced as needs_judge). Every objectively-correct response using
# these words was silently mis-scored. See numeral_scales.csv (NS13-NS15) and _SCALE_WORD_RE.
# ---------------------------------------------------------------------------

# "How many quadrillion make 1 quintillion?" -> 1000.0 quadrillion. Which arab/kharab-ladder pair
# this control item actually asks about is an implementation detail of generate_controls.py's
# pool-index assignment (see next_pool_index there) - re-derive the expected wording/value from
# the item itself rather than hardcoding a prior regeneration's pairing, so this test doesn't
# silently start asserting the wrong pair again the next time the generator's pool bookkeeping
# changes (which is exactly how this test's old "quintillion make 1 sextillion" expectation went
# stale after the repeated-control-prompt fix reassigned which pair BKP-C1-5017 gets).
LADDER_ITEM = CORPUS.controls["BKP-C1-5017"]
_LADDER_UNIT = LADDER_ITEM["gold"]["unit"]
_LADDER_VALUE = int(LADDER_ITEM["gold"]["value"])
_LADDER_HI = LADDER_ITEM["prompt"].rsplit(" ", 1)[-1].rstrip("?")


@pytest.mark.parametrize("make_response,expected", [
    (lambda unit, value, hi: f"{value} {unit}", True),
    (lambda unit, value, hi: f"{value:,} {unit} make 1 {hi}.", True),
    (lambda unit, value, hi: f"{value // 10} {unit}", False),  # wrong magnitude, must not accidentally parse as correct
])
def test_numeric_with_unit_parses_quintillion_scale_word(make_response, expected):
    result = grade(LADDER_ITEM, make_response(_LADDER_UNIT, _LADDER_VALUE, _LADDER_HI))
    assert result.correct is expected, result.reason


@pytest.mark.parametrize("word,power", [("quadrillion", 15), ("quintillion", 18), ("sextillion", 21)])
def test_numeric_extracts_correct_magnitude_for_each_new_scale_word(word, power):
    item = {
        "gold": {"value": float(10 ** power)},
        "answer_type": "numeric",
        "grader": "numeric_v1",
    }
    result = grade(item, f"The answer is 1 {word}.")
    assert result.correct is True, result.reason


def test_numeric_ladder_control_items_are_internally_consistent():
    # Regression guard for the generate_controls.py wraparound bug: for every "How many X make 1
    # Y?" C1 control item, X's magnitude must genuinely be smaller than Y's, and gold.value must
    # equal the real 10^(Y-X) ratio - not a hardcoded constant that silently stopped matching once
    # the cyclic index wrapped back to the start of the ladder.
    scale = {"thousand": 3, "million": 6, "billion": 9, "trillion": 12,
             "quadrillion": 15, "quintillion": 18, "sextillion": 21}
    checked = 0
    for item in ALL_RECORDS:
        m = re.match(r"How many (\w+) make 1 (\w+)\?", item["prompt"])
        if not m or m.group(1) not in scale or m.group(2) not in scale:
            continue
        lo, hi = m.group(1), m.group(2)
        assert scale[lo] < scale[hi], f"{item['id']}: {lo!r} is not smaller than {hi!r}"
        expected = 10 ** (scale[hi] - scale[lo])
        gold_value = item["gold"]["value"]
        assert gold_value == pytest.approx(expected), f"{item['id']}: gold {gold_value} != {expected}"
        checked += 1
    assert checked > 0  # sanity: the parametrization above must actually be exercising items


def test_c7_zip_first_digit_5_is_midwest_not_south():
    # Found auditing the control corpus: this row hardcoded gold="South" for a US ZIP code
    # starting with 5, but digit 5 covers Iowa/Minnesota/Montana/North Dakota/South Dakota/
    # Wisconsin - the upper Midwest, not the South (confirmed against USPS's own first-digit
    # breakdown). Every other digit-5 core item variant produces this same control row, so this
    # is the one place that fact lives.
    item = CORPUS.controls["BKP-C7-5001"]
    assert item["gold"]["value"] == "Midwest"
    result = grade(item, "Midwest")
    assert result.correct is True


def test_c4_shared_regional_name_row_names_two_actually_matching_regions():
    # Found in a manual C4/C5 audit: _REASSIGN_NAMES gives Region A-G seven DISTINCT names, so a
    # bare "two regions share a name, what is it?" premise had no support in the rest of this
    # convention - it named no regions at all and just returned _REASSIGN_NAMES[0]. Now it names
    # Region A and Region H (a region introduced only by this row) so the premise is actually true
    # within the stated convention, mirroring the paired core item's real duplicate (West Bengal
    # and Assam both call their rabi/summer rice crop "boro").
    item = CORPUS.controls["BKP-C4-5044"]
    assert "Region A" in item["prompt"] and "Region H" in item["prompt"]
    assert item["gold"]["value"] == "deepwater"
    # and Region A's name must still match what BKP-C4-5042 (the individual per-region row)
    # independently says Region A is - the whole point of the row is that these two facts agree.
    assert CORPUS.controls["BKP-C4-5042"]["gold"]["value"] == "deepwater"


@pytest.mark.parametrize("rid,expected_month", [("BKP-C4-5085", "June"), ("BKP-C4-5086", "October")])
def test_c4_msp_timing_matches_the_retained_approval_dates(rid, expected_month):
    # Found in the same audit: these rows previously used the Wet/Dry season_definitions window's
    # FIRST month (May/September) as "the month MSP is announced before" - but this same control
    # set separately restates the real approval dates verbatim (28 May 2025 for Wet-crop, 16 Oct
    # 2024 for Dry-crop, in BKP-C4-5082/5083), and 28 May is not before May, nor is 16 Oct before
    # September. The real msp_policy_facts.csv note (kharif->June, rabi->October) is itself a
    # separately-sourced fact, independent of the season window, restated as-is like the dates are.
    item = CORPUS.controls[rid]
    assert item["gold"]["value"] == expected_month


def test_c3_ground_to_cent_conversion_uses_relative_tolerance():
    # Found auditing decimal-place handling: this is the one "ratio" conversion (of many in this
    # subcategory) whose result doesn't land on a whole number - 2400/435.6 = 5.509642... - and
    # its own prompt asks for two-decimal rounding, but generate_c3_items.py's "ratio" override
    # unconditionally used exact tolerance (copied from siblings that DO land on whole numbers).
    # A model correctly rounding to "5.51" as instructed was marked WRONG against the unrounded
    # 6-decimal exact-match gold. Fixed to use relative tolerance like every other rounded-
    # instruction item in this file.
    item = CORPUS.items["BKP-C3-0059"]
    assert item["tolerance"] == {"type": "relative", "value": 0.01}
    assert grade(item, "5.51 cent").correct is True


_ROUND_N_RE = re.compile(r"rounded to (\w+) decimal", re.IGNORECASE)
_ROUND_N_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
                   "1": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6}


def test_no_rounded_instruction_item_has_exact_tolerance_with_unrounded_gold():
    # Corpus-wide guard for the exact BKP-C3-0059 failure mode: a prompt that instructs the model
    # to round its answer to N decimals, paired with exact-match tolerance, where gold itself
    # carries MORE decimal digits than N - meaning a model that correctly follows the rounding
    # instruction gets marked wrong. (Gold carrying FEWER digits than N, e.g. 50.0 for "rounded to
    # two decimals", is fine - that's just a trailing zero Python's float repr drops.)
    offenders = []
    for item in ALL_RECORDS:
        if item["answer_type"] not in ("numeric", "numeric_with_unit"):
            continue
        m = _ROUND_N_RE.search(item["prompt"])
        if not m:
            continue
        n = _ROUND_N_WORDS.get(m.group(1).lower())
        if n is None:
            continue
        tol = item.get("tolerance")
        if tol is not None and tol.get("type") != "exact":
            continue  # relative/absolute tolerance absorbs any rounding difference
        value = item["gold"].get("value")
        if not isinstance(value, (int, float)):
            continue
        s = repr(float(value))
        decimal_places = len(s.split(".")[1]) if "." in s else 0
        if decimal_places > n:
            offenders.append((item["id"], n, value))
    assert offenders == []


# ---------------------------------------------------------------------------
# date / date_range
# ---------------------------------------------------------------------------

DATE_ITEM = CORPUS.items["BKP-C4-0082"]  # gold "2025-05-28"
RANGE_ITEM = CORPUS.items["BKP-C4-0060"]  # gold start=2023-07-01, end=2024-06-30


@pytest.mark.parametrize("text", ["2025-05-28", "28 May 2025", "May 28, 2025", "28-05-2025"])
def test_date_formats_all_parse_to_same_answer(text):
    assert grade(DATE_ITEM, text).correct is True


def test_date_wrong_day():
    assert grade(DATE_ITEM, "2025-05-29").correct is False


def test_date_range_free_text():
    result = grade(RANGE_ITEM, "The crop year 2023-24 runs from 2023-07-01 to 2024-06-30.")
    assert result.correct is True


def test_date_range_wrong_end():
    result = grade(RANGE_ITEM, "From 2023-07-01 to 2024-03-31.")
    assert result.correct is False


# ---------------------------------------------------------------------------
# enum
# ---------------------------------------------------------------------------

ENUM_ITEM = CORPUS.items["BKP-C2-0053"]  # gold "false"


@pytest.mark.parametrize("text", ["False", "No", "no, they differ", "Incorrect"])
def test_enum_boolean_synonyms_accepted(text):
    assert grade(ENUM_ITEM, text).correct is True


def test_enum_opposite_rejected():
    assert grade(ENUM_ITEM, "Yes, true").correct is False


# ---------------------------------------------------------------------------
# string_normalized
# ---------------------------------------------------------------------------

DIGIT_GROUPING_ITEM = CORPUS.items["BKP-C1-0037"]  # gold "12,34,567"
WORD_FORM_ITEM = CORPUS.items["BKP-C1-0109"]  # gold "forty-five thousand six hundred seventy-eight"


def test_digit_grouping_exact_match_required():
    assert grade(DIGIT_GROUPING_ITEM, "12,34,567").correct is True


def test_digit_grouping_international_grouping_rejected():
    # the exact trap this item is testing for - international grouping instead of Indian.
    assert grade(DIGIT_GROUPING_ITEM, "1,234,567").correct is False


def test_digit_grouping_currency_and_whitespace_stripped_but_nothing_else():
    assert grade(DIGIT_GROUPING_ITEM, "₹ 12,34,567").correct is True


def test_word_form_accepts_rephrasing_via_semantic_comparison():
    result = grade(WORD_FORM_ITEM, "The answer is forty five thousand, six hundred and seventy eight.")
    assert result.correct is True


def test_word_form_rejects_wrong_value():
    result = grade(WORD_FORM_ITEM, "forty-five thousand six hundred seventy-nine")
    assert result.correct is False


# ---------------------------------------------------------------------------
# month_set
# ---------------------------------------------------------------------------

MONTH_ITEM = CORPUS.items["BKP-C4-0001"]  # gold months [6, 7]


def test_month_set_names_free_text():
    assert grade(MONTH_ITEM, "Kharif sowing typically happens in June and July.").correct is True


def test_month_set_wrong_months():
    assert grade(MONTH_ITEM, "Kharif sowing happens in October and November.").correct is False


def test_month_set_no_months_found_declines():
    result = grade(MONTH_ITEM, "It varies by crop and region.")
    assert result.correct is None
    assert result.needs_judge is True


# ---------------------------------------------------------------------------
# clarification
# ---------------------------------------------------------------------------

CLARIFICATION_ITEM = CORPUS.items["BKP-C2-0054"]  # maund: Bengal/Punjab/modern-colloquial
# NOTE: this used to be BKP-C2-0046 (seer: trade-customary vs statutory). That item was retired
# from `clarification` to a plain scalar item after correcting the underlying fact - the 1956
# Act's statutory seer (0.933104304 kg, 80 tola) turned out to agree with the trade-customary
# seer (0.9331 kg), not disagree with it.

def test_clarification_hedge_language_correct():
    result = grade(CLARIFICATION_ITEM, "It depends on the convention: Bengal or Punjab.")
    assert result.correct is True
    assert result.extra["single_confident_value"] is False


def test_clarification_naming_two_conventions_correct_without_hedge_words():
    result = grade(
        CLARIFICATION_ITEM,
        "Bengal/standard: 37.324 kg; Punjab: 36.740 kg",
    )
    assert result.correct is True


def test_clarification_single_confident_value_flagged_as_overconfident():
    result = grade(CLARIFICATION_ITEM, "10 maund is 37.324 kg.")
    assert result.correct is False
    assert result.extra["single_confident_value"] is True


def test_clarification_r1_explicit_flag():
    result = grade(CLARIFICATION_ITEM, json.dumps({"clarification_required": True}), regime="R1")
    assert result.correct is True


# ---------------------------------------------------------------------------
# no judge cascade (removed - schema.json x-changelog 0.1.4: 0 judge calls were ever made across
# any run, and no item has ever used grader='judge_v1', so grade() no longer has a judge fallback
# at all. A deterministic decline is now the final verdict, full stop - it stays correct=None for
# score.py's undetermined_rate bucket instead of cascading anywhere else.)
# ---------------------------------------------------------------------------

def test_grade_declined_verdict_is_final_no_judge_cascade(monkeypatch):
    import bkp_eval.graders as graders_mod

    def declines(item, response, regime):
        return graders_mod.GradeResult(correct=None, needs_judge=True, reason="stubbed decline")

    monkeypatch.setitem(graders_mod.DETERMINISTIC_GRADERS, "clarification_v1", declines)
    result = grade(CLARIFICATION_ITEM, "anything")
    assert result.correct is None
    assert result.needs_judge is True

    monkeypatch.setitem(graders_mod.DETERMINISTIC_GRADERS, "numeric_v1", declines)
    result = grade(NUMERIC_ITEM, "anything")
    assert result.correct is None
