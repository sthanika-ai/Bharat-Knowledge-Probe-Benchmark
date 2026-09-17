from __future__ import annotations

import json

import pytest

from bkp_eval.adapters.base import RateLimitedError
from bkp_eval.adapters.mock import MockAdapter
from bkp_eval.items import load_corpus

CORPUS = load_corpus()
ITEM = CORPUS.items["BKP-C1-0001"]  # gold {"value": 500000.0}


def test_oracle_returns_gold_value():
    adapter = MockAdapter(model_id="mock-oracle", accuracy=1.0)
    resp = adapter.complete("Express 5 lakh as a plain number.", item=ITEM)
    assert json.loads(resp.text)["value"] == ITEM["gold"]["value"]
    assert resp.model_id == "mock-oracle"
    assert resp.error is None


def test_zero_accuracy_never_returns_gold_value():
    adapter = MockAdapter(model_id="mock-wrong", accuracy=0.0)
    resp = adapter.complete("Express 5 lakh as a plain number.", item=ITEM)
    assert json.loads(resp.text)["value"] != ITEM["gold"]["value"]


def test_same_prompt_same_seed_is_deterministic():
    a = MockAdapter(accuracy=0.5, seed=7).complete("some prompt", item=ITEM)
    b = MockAdapter(accuracy=0.5, seed=7).complete("some prompt", item=ITEM)
    assert a.text == b.text


def test_different_seed_can_differ():
    seeds_results = {
        MockAdapter(accuracy=0.5, seed=s).complete("some prompt", item=ITEM).text
        for s in range(20)
    }
    assert len(seeds_results) > 1  # not every seed collides to the same outcome


def test_flaky_raises_ratelimit_once_then_succeeds():
    adapter = MockAdapter(model_id="mock-flaky", accuracy=1.0, flaky_rate=1.0)  # always flakes first
    with pytest.raises(RateLimitedError):
        adapter.complete("a prompt", item=ITEM)
    # second attempt at the SAME prompt succeeds - the adapter tracks attempts per prompt key
    resp = adapter.complete("a prompt", item=ITEM)
    assert resp.error is None


def test_no_item_returns_empty_text():
    resp = MockAdapter().complete("a prompt", item=None)
    assert resp.text == ""
