"""score.py tests: small hand-built cases pin down exact arithmetic for each metric; the fixture
test proves the whole pipeline (real corpus + a results log) produces sane, real-looking numbers
end to end - see scripts/build_demo_results.py for how the fixture was made.
"""
from __future__ import annotations

from pathlib import Path

from bkp_eval.graders import GradeResult
from bkp_eval.items import load_corpus
from bkp_eval.score import (
    GradedRecord,
    per_item_core_accuracy,
    per_item_core_accuracy_by_category,
    score,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURE = REPO_ROOT / "tests" / "fixtures" / "demo_results.jsonl"
CORPUS = load_corpus()


def _core_rec(item_id: str, category: str, correct: bool, regime: str = "R1") -> GradedRecord:
    item = {"id": item_id, "category": category}
    return GradedRecord(item=item, response="", regime=regime, result=GradeResult(correct=correct))


def test_score_on_demo_fixture_end_to_end():
    reports = score(FIXTURE, corpus=CORPUS)
    assert set(reports.keys()) == {"demo-model-v0"}
    report = reports["demo-model-v0"]

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
