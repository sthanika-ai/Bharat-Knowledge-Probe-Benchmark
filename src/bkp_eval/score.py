"""score.py - turns a results log (model responses against BKP-500 items) into a metrics table:
Bharat Score, per-category accuracy, and - the piece that actually measures locale competence
as distinct from raw arithmetic - Locale Gap Delta.

Input format: a JSONL file, one object per line - exactly what runner.py writes, though building
one by hand (or from something else you used to query models) only needs these fields:
  item_id       - str, must match an id in data/items/*.jsonl or data/controls/controls.jsonl
  response      - str, the raw model output (JSON per R1, free text per R2 - see graders/_common.py)
  model         - str, optional, defaults to "unknown" - one file can hold several models' runs
  regime        - "R1" | "R2", optional, defaults to "R2"
  sample_index  - int, optional, defaults to 0 - which of the n repeated samples this is; only
                  matters for locale_gap() (pairs same-sample-index responses) and consistency()

Run as `python -m bkp_eval.score path/to/results.jsonl`.
"""
from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from bkp_eval.graders import GradeResult, grade
from bkp_eval.items import Corpus, load_corpus

CATEGORIES = [
    "C1_numerals", "C2_weights_volumes", "C3_land_units", "C4_seasons",
    "C5_fiscal_year", "C6_schemes", "C7_identifiers",
]


@dataclass
class GradedRecord:
    item: dict
    response: str
    regime: str
    result: GradeResult
    sample_index: int = 0


def load_results(path: Path | str) -> list[dict]:
    records = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def grade_results(results: list[dict], corpus: Corpus) -> dict[str, list[GradedRecord]]:
    """Grade every result row and group the graded records by `model`."""
    by_model: dict[str, list[GradedRecord]] = defaultdict(list)
    for row in results:
        item = corpus.all_by_id.get(row["item_id"])
        if item is None:
            raise KeyError(f"results row references unknown item_id {row['item_id']!r}")
        response = row["response"]
        regime = row.get("regime", "R2")
        result = grade(item, response, regime)
        model = row.get("model", "unknown")
        by_model[model].append(GradedRecord(
            item=item, response=response, regime=regime, result=result,
            sample_index=row.get("sample_index", 0),
        ))
    return by_model


def _accuracy(records: list[GradedRecord]) -> tuple[float, int]:
    """(accuracy, n) over `records`, treating undetermined (correct=None) as not-correct.

    Refusals are reported separately, never hidden inside
    accuracy, by never shrinking the denominator to make an ungradeable response disappear.
    Pair this with undetermined_rate()/refusal_rate() below so a high share of either is visible
    rather than quietly folded into an accuracy number that looks better than it should.
    """
    if not records:
        return 0.0, 0
    correct = sum(1 for r in records if r.result.correct is True)
    return correct / len(records), len(records)


def _rate(records: list[GradedRecord], predicate) -> float:
    if not records:
        return 0.0
    return sum(1 for r in records if predicate(r)) / len(records)


def per_item_core_accuracy(records: list[GradedRecord]) -> dict[str, float]:
    """item_id -> accuracy, averaged across every one of that item's graded records (repeated
    samples AND prompt regimes pooled together), core items only.

    This is the unit both paired_bootstrap_diff (comparing two models on identical items) and
    any bootstrap CI over these records (bootstrap_ci / bootstrap_ci_macro) should resample -
    NOT the raw per-response rows. A model's n=3 repeated samples of the same item are correlated
    measurements of one underlying fact, not 3 independent items; feeding raw per-response 0/1
    values into a bootstrap would silently treat them as if they were, inflating the effective
    sample size and understating the true uncertainty (pseudoreplication). This matters more than
    usual here: BKP-500 runs at temperature 0 by default, and an audit of real run data found
    ~95% of same-item repeated samples come back byte-identical - the 3 "samples" are overwhelmingly
    not independent draws at all, so aggregating them into one point per item before any
    statistical test is not just good practice, it's close to necessary.
    """
    per_item: dict[str, list[float]] = defaultdict(list)
    for r in records:
        if r.item["is_control"]:
            continue
        per_item[r.item["id"]].append(1.0 if r.result.correct else 0.0)
    return {item_id: statistics.fmean(v) for item_id, v in per_item.items()}


def per_item_core_accuracy_by_category(records: list[GradedRecord]) -> dict[str, list[float]]:
    """category -> list of item-level mean accuracies (one entry per distinct core item in that
    category, already aggregated across repeats/regimes) - bootstrap_ci_macro's expected input
    for a pseudoreplication-free Bharat Score CI. See per_item_core_accuracy's docstring for why
    aggregation has to happen before, not after, bootstrapping.
    """
    per_item = per_item_core_accuracy(records)
    category_of: dict[str, str] = {}
    for r in records:
        if not r.item["is_control"]:
            category_of[r.item["id"]] = r.item["category"]
    by_category: dict[str, list[float]] = {cat: [] for cat in CATEGORIES}
    for item_id, acc in per_item.items():
        by_category[category_of[item_id]].append(acc)
    return by_category


def per_pair_locale_delta(pairs: list[tuple[GradedRecord, GradedRecord]]) -> dict[str, float]:
    """core item_id -> mean (control_correct - india_correct), averaged across every sample/regime
    that item-pair was scored under. The Locale Gap Delta analogue of per_item_core_accuracy: a
    matched pair's repeated samples are correlated measurements of the same pair, not independent
    pairs, so this is the unit to bootstrap a CI over for Locale Gap Delta - not the raw per-sample
    deltas locale_gap() itself averages over.
    """
    per_item: dict[str, list[float]] = defaultdict(list)
    for india_rec, control_rec in pairs:
        c = 1 if control_rec.result.correct is True else 0
        i = 1 if india_rec.result.correct is True else 0
        per_item[india_rec.item["id"]].append(c - i)
    return {item_id: statistics.fmean(v) for item_id, v in per_item.items()}


def bharat_score(core_records: list[GradedRecord]) -> dict:
    """Macro-mean of C1-C7 accuracy, equal category weight - resists a model
    gaming the headline by being strong only on the largest category.
    """
    per_category = {}
    for cat in CATEGORIES:
        cat_records = [r for r in core_records if r.item["category"] == cat]
        acc, n = _accuracy(cat_records)
        per_category[cat] = {"accuracy": acc, "n": n}
    scored = [v["accuracy"] for v in per_category.values() if v["n"] > 0]
    macro_mean = statistics.fmean(scored) if scored else 0.0
    return {"bharat_score": macro_mean, "per_category": per_category}


def locale_gap(pairs: list[tuple[GradedRecord, GradedRecord]]) -> dict:
    """Locale Gap Delta = acc(control) - acc(India-framed), computed only over
    matched pairs where BOTH twins have a graded response.

    Also reports std_delta, the per-pair standard deviation of (control_correct - india_correct)
    - not just the mean - since a consistent 0.3-point gap across hundreds of pairs and a mean
    of 0.3 driven by a handful of outlier pairs look identical as a single number but say very
    different things about whether the effect is real and general. See the paired-design
    discussion this function operationalizes.
    """
    if not pairs:
        return {"n_pairs": 0, "mean_delta": None, "std_delta": None,
                "accuracy_control": None, "accuracy_india": None}
    per_pair_delta = []
    control_correct = []
    india_correct = []
    for india_rec, control_rec in pairs:
        c = 1 if control_rec.result.correct is True else 0
        i = 1 if india_rec.result.correct is True else 0
        control_correct.append(c)
        india_correct.append(i)
        per_pair_delta.append(c - i)
    return {
        "n_pairs": len(pairs),
        "accuracy_control": statistics.fmean(control_correct),
        "accuracy_india": statistics.fmean(india_correct),
        "mean_delta": statistics.fmean(per_pair_delta),
        "std_delta": statistics.pstdev(per_pair_delta) if len(per_pair_delta) > 1 else 0.0,
    }


def oom_error_rate(records: list[GradedRecord]) -> dict:
    """Share of ALL numeric/numeric_with_unit items (not just the wrong ones) answered off by
    >=10x - the "lakh/crore catastrophe rate."
    """
    numeric_records = [r for r in records if r.item["answer_type"] in ("numeric", "numeric_with_unit")]
    rate = _rate(numeric_records, lambda r: r.result.extra.get("oom_error") is True)
    return {"rate": rate, "n": len(numeric_records)}


def unit_discipline(records: list[GradedRecord]) -> dict:
    """Share of numeric_with_unit answers carrying a correct explicit unit, independent of
    whether the numeric value itself was right - a silent unit drop is its own failure mode.
    """
    unit_records = [r for r in records if r.item["answer_type"] == "numeric_with_unit"]
    rate = _rate(unit_records, lambda r: r.result.extra.get("has_explicit_unit") is True)
    return {"rate": rate, "n": len(unit_records)}


def ambiguity_handling(records: list[GradedRecord]) -> dict:
    """Accuracy on `clarification`-type items - rewards "it depends on the convention/state"."""
    clar_records = [r for r in records if r.item["answer_type"] == "clarification"]
    acc, n = _accuracy(clar_records)
    return {"accuracy": acc, "n": n}


def overconfidence(records: list[GradedRecord]) -> dict:
    """Share of `clarification`-type items answered with a single confident scalar and no hedge
    - the more damning inverse framing of Ambiguity Handling.
    """
    clar_records = [r for r in records if r.item["answer_type"] == "clarification"]
    rate = _rate(clar_records, lambda r: r.result.extra.get("single_confident_value") is True)
    return {"rate": rate, "n": len(clar_records)}


def refusal_rate(records: list[GradedRecord]) -> dict:
    return {"rate": _rate(records, lambda r: r.result.is_refusal), "n": len(records)}


def undetermined_rate(records: list[GradedRecord]) -> dict:
    """Share of responses the deterministic grader (and judge cascade, where eligible) could not
    rule on at all - distinct from a wrong answer, and worth watching per-model: a model that
    free-associates instead of answering will show up here, not as low accuracy.
    """
    return {"rate": _rate(records, lambda r: r.result.correct is None), "n": len(records)}


def consistency(records: list[GradedRecord]) -> dict:
    """Agreement across repeated samples of the SAME item+regime (the "agreement
    across 3 samples" half of the Consistency metric - the en<->hi paired-slice half needs a
    Hindi-translated item set that doesn't exist in this corpus yet, so it isn't computed here).

    An (item_id, regime) group with only one sample can't say anything about consistency and is
    excluded from both the numerator and denominator, rather than counted as either "consistent"
    or "inconsistent" by default.
    """
    groups: dict[tuple[str, str], list[bool | None]] = defaultdict(list)
    for r in records:
        groups[(r.item["id"], r.regime)].append(r.result.correct)
    multi_sample_groups = {k: v for k, v in groups.items() if len(v) >= 2}
    if not multi_sample_groups:
        return {"rate": None, "n_items": 0}
    agree = sum(1 for verdicts in multi_sample_groups.values() if len(set(verdicts)) == 1)
    return {"rate": agree / len(multi_sample_groups), "n_items": len(multi_sample_groups)}


def score_model(records: list[GradedRecord], corpus: Corpus) -> dict:
    core_records = [r for r in records if not r.item["is_control"]]

    by_id: dict[str, list[GradedRecord]] = defaultdict(list)
    for r in records:
        by_id[r.item["id"]].append(r)
    for recs in by_id.values():
        recs.sort(key=lambda r: r.sample_index)

    # Pair same-sample-index responses (core sample 0 <-> control sample 0, etc.) rather than
    # collapsing each item to a single record first - this is what makes n_pairs scale with
    # n_samples and keeps each sample an independent paired comparison for locale_gap()'s std_delta.
    matched_pairs = [
        (core_rec, control_rec)
        for core, control in corpus.pairs()
        for core_rec, control_rec in zip(by_id.get(core["id"], []), by_id.get(control["id"], []))
    ]

    return {
        **bharat_score(core_records),
        "locale_gap": locale_gap(matched_pairs),
        "oom_error_rate": oom_error_rate(records),
        "unit_discipline": unit_discipline(records),
        "ambiguity_handling": ambiguity_handling(records),
        "overconfidence": overconfidence(records),
        "refusal_rate": refusal_rate(records),
        "undetermined_rate": undetermined_rate(records),
        "consistency": consistency(records),
    }


def score(results_path: Path | str, corpus: Corpus | None = None) -> dict[str, dict]:
    corpus = corpus or load_corpus()
    results = load_results(results_path)
    by_model = grade_results(results, corpus)
    return {model: score_model(records, corpus) for model, records in by_model.items()}


def _format_report(model: str, report: dict) -> str:
    lg = report["locale_gap"]
    lines = [f"=== {model} ===", f"Bharat Score:        {report['bharat_score']:.1%}"]
    for cat, v in report["per_category"].items():
        lines.append(f"  {cat:<22} {v['accuracy']:.1%}  (n={v['n']})")
    if lg["n_pairs"]:
        lines.append(
            f"Locale Gap Delta:     {lg['mean_delta']:+.1%}  (std={lg['std_delta']:.1%}, "
            f"n_pairs={lg['n_pairs']}, control={lg['accuracy_control']:.1%}, "
            f"india={lg['accuracy_india']:.1%})"
        )
    else:
        lines.append("Locale Gap Delta:     no matched pairs with responses on both twins")
    lines += [
        f"OOM Error Rate:       {report['oom_error_rate']['rate']:.1%}  (n={report['oom_error_rate']['n']})",
        f"Unit Discipline:      {report['unit_discipline']['rate']:.1%}  (n={report['unit_discipline']['n']})",
        f"Ambiguity Handling:   {report['ambiguity_handling']['accuracy']:.1%}  (n={report['ambiguity_handling']['n']})",
        f"Overconfidence:       {report['overconfidence']['rate']:.1%}  (n={report['overconfidence']['n']})",
        f"Refusal rate:         {report['refusal_rate']['rate']:.1%}  (n={report['refusal_rate']['n']})",
        f"Undetermined rate:    {report['undetermined_rate']['rate']:.1%}  (n={report['undetermined_rate']['n']})",
        (
            f"Consistency:          {report['consistency']['rate']:.1%}  (n={report['consistency']['n_items']})"
            if report["consistency"]["rate"] is not None
            else "Consistency:          not computable - no item+regime has more than 1 sample"
        ),
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("results", type=Path, help="path to a results .jsonl file")
    args = parser.parse_args()

    reports = score(args.results)
    for model, report in reports.items():
        print(_format_report(model, report))
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
