"""runner.py tests - all against MockAdapter or tiny in-test adapter doubles, no network calls
and no real sleeping (sleep_fn is always replaced with a recording no-op).
"""
from __future__ import annotations

import json

from bkp_eval.adapters.base import AdapterError, ModelAdapter, RateLimitedError
from bkp_eval.adapters.mock import MockAdapter
from bkp_eval.items import load_corpus
from bkp_eval.runner import RunConfig, run
from bkp_eval.score import score

CORPUS = load_corpus()
ITEM_IDS = ["BKP-C1-0001", "BKP-C1-0002", "BKP-C1-5001"]  # two core items + one control


class AlwaysRateLimitedAdapter(ModelAdapter):
    """Never succeeds - every call raises RateLimitedError, so max_retries is always exhausted."""

    model_id = "always-429"

    def complete(self, prompt, *, system=None, temperature=0.0, max_tokens=512, item=None):
        raise RateLimitedError("always limited")


class AlwaysBrokenAdapter(ModelAdapter):
    """A non-rate-limit failure - should NOT be retried at all."""

    model_id = "always-broken"

    def __init__(self):
        self.call_count = 0

    def complete(self, prompt, *, system=None, temperature=0.0, max_tokens=512, item=None):
        self.call_count += 1
        raise AdapterError("the model service is down")


def _read_lines(path):
    with open(path) as fh:
        return [json.loads(line) for line in fh if line.strip()]


def test_run_writes_expected_number_of_rows(tmp_path):
    out_path = tmp_path / "results.jsonl"
    config = RunConfig(regimes=("R1", "R2"), n_samples=2)
    stats = run(CORPUS, MockAdapter(accuracy=1.0), ITEM_IDS, out_path, config=config, sleep_fn=lambda s: None)

    rows = _read_lines(out_path)
    assert len(rows) == len(ITEM_IDS) * 2 * 2  # items * regimes * n_samples
    assert stats.calls_made == len(rows)
    assert stats.calls_skipped_resumed == 0
    assert stats.errors == 0


def test_run_is_resumable_and_never_duplicates_rows(tmp_path):
    out_path = tmp_path / "results.jsonl"
    config = RunConfig(regimes=("R1",), n_samples=1)

    stats1 = run(CORPUS, MockAdapter(accuracy=1.0), ITEM_IDS, out_path, config=config, sleep_fn=lambda s: None)
    assert stats1.calls_made == len(ITEM_IDS)

    # "resume" - same call again, same output file
    stats2 = run(CORPUS, MockAdapter(accuracy=1.0), ITEM_IDS, out_path, config=config, sleep_fn=lambda s: None)
    assert stats2.calls_made == 0
    assert stats2.calls_skipped_resumed == len(ITEM_IDS)

    rows = _read_lines(out_path)
    assert len(rows) == len(ITEM_IDS)  # not doubled


def test_run_resumes_partial_progress_from_a_prior_interrupted_run(tmp_path):
    out_path = tmp_path / "results.jsonl"
    config = RunConfig(regimes=("R1",), n_samples=1)

    # simulate a partial prior run that only covered the first item
    run(CORPUS, MockAdapter(accuracy=1.0), ITEM_IDS[:1], out_path, config=config, sleep_fn=lambda s: None)
    # resume with the FULL item list
    stats = run(CORPUS, MockAdapter(accuracy=1.0), ITEM_IDS, out_path, config=config, sleep_fn=lambda s: None)

    assert stats.calls_skipped_resumed == 1
    assert stats.calls_made == len(ITEM_IDS) - 1
    assert len(_read_lines(out_path)) == len(ITEM_IDS)


def test_flaky_adapter_retries_and_eventually_succeeds(tmp_path):
    out_path = tmp_path / "results.jsonl"
    config = RunConfig(regimes=("R1",), n_samples=1, max_retries=3)
    delays = []

    stats = run(
        CORPUS, MockAdapter(accuracy=1.0, flaky_rate=1.0), ITEM_IDS, out_path,
        config=config, sleep_fn=delays.append,
    )

    assert stats.rate_limit_retries == len(ITEM_IDS)  # each item flaked exactly once, then succeeded
    assert stats.errors == 0
    assert delays  # backoff actually slept (recorded), just not for real


def test_persistent_rate_limit_gives_up_after_max_retries_and_logs_error(tmp_path):
    out_path = tmp_path / "results.jsonl"
    config = RunConfig(regimes=("R1",), n_samples=1, max_retries=3, backoff_base_s=0.01)
    delays = []

    stats = run(CORPUS, AlwaysRateLimitedAdapter(), ITEM_IDS[:1], out_path, config=config, sleep_fn=delays.append)

    assert stats.errors == 1
    assert stats.rate_limit_retries == config.max_retries
    assert len(delays) == config.max_retries
    row = _read_lines(out_path)[0]
    assert row["error"] is not None
    assert "gave up" in row["error"]


def test_non_rate_limit_error_is_not_retried(tmp_path):
    adapter = AlwaysBrokenAdapter()
    sleeps = []
    stats = run(
        CORPUS, adapter, ITEM_IDS[:1], tmp_path / "results.jsonl",
        config=RunConfig(regimes=("R1",), n_samples=1), sleep_fn=sleeps.append,
    )
    assert adapter.call_count == 1  # no retry loop for a non-rate-limit AdapterError
    assert stats.errors == 1
    assert sleeps == []  # never backed off - a plain AdapterError isn't retryable


def test_runner_output_is_directly_consumable_by_score(tmp_path):
    out_path = tmp_path / "results.jsonl"
    config = RunConfig(regimes=("R1", "R2"), n_samples=1)
    run(CORPUS, MockAdapter(model_id="demo", accuracy=0.7, seed=3), ITEM_IDS, out_path, config=config, sleep_fn=lambda s: None)

    reports = score(out_path, corpus=CORPUS)
    assert "demo" in reports
    assert 0.0 <= reports["demo"]["bharat_score"] <= 1.0
