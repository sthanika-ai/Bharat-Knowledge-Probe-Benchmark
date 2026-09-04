"""A zero-cost, zero-network adapter for exercising runner.py's plumbing (sampling, resumability,
retry/backoff) without an API key or a real bill. Real provider adapters (anthropic.py, openai.py)
are implemented the same shape but were never exercised live in this environment - no provider
package/API key is configured here. Everything in tests/ runs against this adapter instead.
"""
from __future__ import annotations

import hashlib
import json
import random
import time

from bkp_eval.adapters.base import AdapterResponse, ModelAdapter, RateLimitedError


def _stable_seed(*parts: str) -> int:
    """A hash seed stable across processes - Python's builtin hash() is salted per-process for
    strings, which would make "the same prompt" flake/score differently every run.
    """
    digest = hashlib.sha256("||".join(parts).encode()).hexdigest()
    return int(digest[:16], 16)


def _gold_as_r1_json(item: dict, correct: bool, rng: random.Random) -> str:
    """A plausible R1-shaped response: the exact gold value when `correct`, a clearly-wrong-but-
    same-shape value otherwise. Deliberately reuses only `item["gold"]`, never any grader
    internals, so this stays an honest black-box "model" rather than one that can see its own
    answer key mid-grading.
    """
    gold = item["gold"]
    answer_type = item["answer_type"]

    if answer_type in ("numeric", "numeric_with_unit"):
        value = gold["value"]
        if not correct:
            value = value * rng.choice([10, 100, 0.1]) if value else 1.0
        payload = {"value": value}
        if answer_type == "numeric_with_unit":
            payload["unit"] = gold["unit"]
        return json.dumps(payload)
    if answer_type == "date":
        return json.dumps({"value": gold["value"] if correct else "1900-01-01"})
    if answer_type == "date_range":
        return json.dumps(gold if correct else {"start": "1900-01-01", "end": "1900-01-02"})
    if answer_type == "month_set":
        months = gold["months"] if correct else [((m % 12) + 1) for m in gold["months"]]
        return json.dumps({"months": months})
    if answer_type == "clarification":
        return json.dumps({"clarification_required": True if correct else False})
    # enum, string_normalized
    value = gold["value"] if correct else f"{gold['value']}-wrong"
    return json.dumps({"value": value})


class MockAdapter(ModelAdapter):
    def __init__(
        self,
        model_id: str = "mock-oracle",
        accuracy: float = 1.0,
        flaky_rate: float = 0.0,
        seed: int = 0,
    ):
        """accuracy: fraction of (item, call) pairs answered with the gold value. flaky_rate:
        fraction of prompts that raise RateLimitedError on their FIRST attempt only, succeeding
        on retry - so runner.py's backoff path gets exercised by something real, not asserted
        in the abstract.
        """
        self.model_id = model_id
        self.accuracy = accuracy
        self.flaky_rate = flaky_rate
        self.seed = seed
        self._attempts: dict[str, int] = {}

    def complete(
        self, prompt: str, *, system: str | None = None, temperature: float = 0.0,
        max_tokens: int = 512, item: dict | None = None,
    ) -> AdapterResponse:
        start = time.monotonic()
        key = f"{prompt}||{system or ''}"
        self._attempts[key] = self._attempts.get(key, 0) + 1
        is_first_attempt = self._attempts[key] == 1

        # Deterministic test-fixture PRNG (seeded from a stable hash): decides whether the mock
        # simulates a flake/wrong answer for test coverage, never anything security-sensitive.
        seed = _stable_seed("flake", str(self.seed), key)
        flake_roll = random.Random(seed).random()  # nosec B311
        if is_first_attempt and flake_roll < self.flaky_rate:
            raise RateLimitedError("mock: simulated 429", retry_after=0.0)

        # Same as above: deterministic mock-response generation, no crypto/security use.
        rng = random.Random(_stable_seed("answer", str(self.seed), key))  # nosec B311
        if item is None:
            text = ""
        else:
            correct = rng.random() < self.accuracy
            text = _gold_as_r1_json(item, correct, rng)

        return AdapterResponse(
            text=text,
            model_id=self.model_id,
            latency_s=time.monotonic() - start,
            input_tokens=len(prompt.split()),
            output_tokens=len(text.split()),
            raw_request={"prompt": prompt, "system": system, "temperature": temperature},
            raw_response={"text": text},
        )
