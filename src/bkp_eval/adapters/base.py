"""The uniform adapter interface every provider (and the mock) implements - one per provider,
uniform interface. runner.py only ever talks to this interface, never to a provider SDK directly,
so adding a new provider never touches runner.py.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


class AdapterError(Exception):
    """Base class for every error an adapter raises on purpose."""


class RateLimitedError(AdapterError):
    """The provider asked us to slow down. `retry_after` is the provider's own hint in seconds,
    when it gave one - runner.py falls back to exponential backoff if it didn't.
    """

    def __init__(self, message: str = "rate limited", retry_after: float | None = None):
        super().__init__(message)
        self.retry_after = retry_after


@dataclass
class AdapterResponse:
    """What every adapter call returns, success or failure - always logged raw, for replication,
    never just the extracted text.
    """

    text: str
    model_id: str
    latency_s: float
    input_tokens: int | None = None
    output_tokens: int | None = None
    raw_request: dict | None = None
    raw_response: dict | None = None
    error: str | None = None


class ModelAdapter(ABC):
    """One instance = one pinned model id. `complete()` is called once per (item, regime, sample) -
    runner.py owns retrying, rate-limit backoff, and sampling count; an adapter just makes one call.

    `item` is passed through for adapters that can use it (the mock does, to fabricate a
    plausible answer); real provider adapters ignore it and only use `prompt`/`system`.
    """

    model_id: str

    @abstractmethod
    def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 512,
        item: dict | None = None,
    ) -> AdapterResponse:
        raise NotImplementedError
