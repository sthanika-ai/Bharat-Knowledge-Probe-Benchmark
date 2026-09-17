"""score.py tests: small hand-built cases pin down exact arithmetic for each metric; the fixture
test proves the whole pipeline (real corpus + a results log) produces a sane, real-looking
Locale Gap Delta end to end - see scripts/build_demo_results.py for how the fixture was made.
"""
from __future__ import annotations

from pathlib import Path

from bkp_eval.graders import GradeResult
from bkp_eval.items import load_corpus
from bkp_eval.score import (
    GradedRecord,
    locale_gap,
    per_item_core_accuracy,
    per_item_core_accuracy_by_category,
    per_pair_locale_delta,
    score,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURE = REPO_ROOT / "tests" / "fixtures" / "demo_results.jsonl"
CORPUS = load_corpus()


def _rec(item_id: str, correct: bool) -> GradedRecord:
    return GradedRecord(item={"id": item_id}, response="", regime="R1", result=GradeResult(correct=correct))


def _core_rec(item_id: str, category: str, correct: bool, regime: str = "R1") -> GradedRecord:
    item = {"id": item_id, "category": category, "is_control": False}
    return GradedRecord(item=item, response="", regime=regime, result=GradeResult(correct=correct))


def _control_rec(item_id: str, correct: bool, regime: str = "R1") -> GradedRecord:
    item = {"id": item_id, "category": "C1_numerals", "is_control": True}
    return GradedRecord(item=item, response="", regime=regime, result=GradeResult(correct=correct))


def test_locale_gap_matches_hand_computed_example():
    # Worked example shape: control strong, india weak.
    pairs = [
        (_rec("india-1", False), _rec("control-1", True)),
        (_rec("india-2", False), _rec("control-2", True)),
        (_rec("india-3", True), _rec("control-3", True)),
        (_rec("india-4", False), _rec("control-4", False)),
    ]
    result = locale_gap(pairs)
    assert result["n_pairs"] == 4
    assert result["accuracy_control"] == 0.75
    assert result["accuracy_india"] == 0.25
    assert result["mean_delta"] == 0.5
    # per-pair deltas are [1, 1, 0, 0] -> population stdev
    assert round(result["std_delta"], 4) == 0.5


def test_locale_gap_empty_pairs():
    result = locale_gap([])
    assert result["n_pairs"] == 0
    assert result["mean_delta"] is None


def test_score_on_demo_fixture_end_to_end():
    reports = score(FIXTURE, corpus=CORPUS)
    assert set(reports.keys()) == {"demo-model-v0"}
    report = reports["demo-model-v0"]

    lg = report["locale_gap"]
    assert lg["n_pairs"] == len(CORPUS.pairs())
    # The fixture was built with control_accuracy=0.92, india_accuracy=0.55 - the observed
    # Delta should land clearly in between "no effect" and "every pair flipped".
    assert 0.25 < lg["mean_delta"] < 0.55
    assert lg["accuracy_control"] > lg["accuracy_india"]
    assert lg["std_delta"] > 0  # not every pair moved identically - confirms it's not a constant

    assert report["oom_error_rate"]["rate"] > 0  # the fixture deliberately injects some
    assert 0 < report["bharat_score"] < 1
    assert report["refusal_rate"]["rate"] == 0  # the fixture never emits a refusal-shaped response


def test_results_row_with_unknown_item_id_raises():
    import pytest

    from bkp_eval.score import grade_results

    with pytest.raises(KeyError):
        grade_results([{"item_id": "BKP-NOPE-9999", "response": "1"}], CORPUS)


def test_load_results_reads_jsonl(tmp_path):
    from bkp_eval.score import load_results

    path = tmp_path / "r.jsonl"
    path.write_text('{"item_id": "a", "response": "1"}\n\n{"item_id": "b", "response": "2"}\n')
    rows = load_results(path)
    assert rows == [{"item_id": "a", "response": "1"}, {"item_id": "b", "response": "2"}]


def test_per_item_core_accuracy_averages_repeated_samples_of_the_same_item():
    # 3 repeated samples of the same item (2 correct, 1 wrong) must collapse to ONE point (2/3),
    # not be treated as 3 separate items - the pseudoreplication fix this function exists for.
    records = [
        _core_rec("i-1", "C1_numerals", True),
        _core_rec("i-1", "C1_numerals", True),
        _core_rec("i-1", "C1_numerals", False),
    ]
    result = per_item_core_accuracy(records)
    assert result == {"i-1": 2 / 3}


def test_per_item_core_accuracy_pools_across_regimes_too():
    records = [
        _core_rec("i-1", "C1_numerals", True, regime="R1"),
        _core_rec("i-1", "C1_numerals", False, regime="R2"),
    ]
    result = per_item_core_accuracy(records)
    assert result == {"i-1": 0.5}


def test_per_item_core_accuracy_excludes_control_items():
    records = [
        _core_rec("i-1", "C1_numerals", True),
        _control_rec("ctl-1", False),
    ]
    result = per_item_core_accuracy(records)
    assert result == {"i-1": 1.0}


def test_per_item_core_accuracy_empty():
    assert per_item_core_accuracy([]) == {}


def test_per_item_core_accuracy_by_category_buckets_one_entry_per_item_not_per_row():
    records = [
        # item a: 3 repeats, all correct -> one entry of 1.0 in C1
        _core_rec("a", "C1_numerals", True),
        _core_rec("a", "C1_numerals", True),
        _core_rec("a", "C1_numerals", True),
        # item b: 2 repeats, one right one wrong -> one entry of 0.5 in C2
        _core_rec("b", "C2_weights_volumes", True),
        _core_rec("b", "C2_weights_volumes", False),
    ]
    result = per_item_core_accuracy_by_category(records)
    assert result["C1_numerals"] == [1.0]  # one item, not three rows
    assert result["C2_weights_volumes"] == [0.5]  # one item, not two rows
    # every other category present with an empty list, not missing - bootstrap_ci_macro expects
    # every category key to exist even when a model has zero graded items in it
    assert result["C3_land_units"] == []


def test_per_pair_locale_delta_averages_repeated_samples_of_the_same_pair():
    # same item-pair scored under 3 repeated samples: control right every time, india right once
    # -> per-sample deltas [1, 1, -1]... wait, deltas are (control_correct - india_correct):
    # sample 1: control=1, india=1 -> delta 0; sample 2: control=1, india=0 -> delta 1;
    # sample 3: control=1, india=0 -> delta 1. Mean should be (0+1+1)/3, one point for the pair.
    pairs = [
        (_rec("india-1", True), _rec("control-1", True)),
        (_rec("india-1", False), _rec("control-1", True)),
        (_rec("india-1", False), _rec("control-1", True)),
    ]
    result = per_pair_locale_delta(pairs)
    assert result == {"india-1": 2 / 3}


def test_per_pair_locale_delta_empty():
    assert per_pair_locale_delta([]) == {}
