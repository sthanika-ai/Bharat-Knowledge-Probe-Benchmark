"""Adapter registry: maps a model id prefix to the adapter class that serves it.

Not exhaustive - several providers (India-built: Sarvam, Krutrim, BharatGPT, ...) don't have a
registered adapter yet, and are served via `LocalAPIAdapter`/`HFTransformersAdapter` instead. Call
register_adapter() to add one without touching this file. get_adapter() falls back to MockAdapter
for anything unregistered rather than raising, so a pipeline smoke test never needs every
provider wired up first.
"""
from __future__ import annotations

from bkp_eval.adapters.base import AdapterError, AdapterResponse, ModelAdapter, RateLimitedError
from bkp_eval.adapters.mock import MockAdapter

__all__ = [
    "AdapterError", "AdapterResponse", "ModelAdapter", "RateLimitedError", "MockAdapter",
    "get_adapter", "register_adapter",
]

_FACTORIES: dict[str, type[ModelAdapter]] = {}


def register_adapter(prefix: str, adapter_cls: type[ModelAdapter]) -> None:
    _FACTORIES[prefix] = adapter_cls


def get_adapter(model_id: str, **kwargs) -> ModelAdapter:
    for prefix, cls in _FACTORIES.items():
        if model_id.startswith(prefix):
            return cls(model_id=model_id, **kwargs)
    return MockAdapter(model_id=model_id, **kwargs)


def _register_known_providers() -> None:
    from bkp_eval.adapters.anthropic_adapter import AnthropicAdapter
    from bkp_eval.adapters.openai_adapter import OpenAIAdapter

    register_adapter("claude-", AnthropicAdapter)
    register_adapter("gpt-", OpenAIAdapter)
    register_adapter("o1-", OpenAIAdapter)


_register_known_providers()
