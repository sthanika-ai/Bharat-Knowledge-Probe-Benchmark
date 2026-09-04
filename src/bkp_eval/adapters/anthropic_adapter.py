"""Anthropic provider adapter (Messages API).

Not exercised against a live model in this environment: the `anthropic` package isn't installed
here and no ANTHROPIC_API_KEY is configured. The SDK import is deliberately deferred to
_get_client()/complete() so this module still imports cleanly (and is registerable) without the
package present - only actually calling complete() requires it. Treat this as implemented-but-
unverified until it's run once against a real key.
"""
from __future__ import annotations

import os
import time

from bkp_eval.adapters.base import AdapterError, AdapterResponse, ModelAdapter, RateLimitedError


class AnthropicAdapter(ModelAdapter):
    def __init__(self, model_id: str, api_key: str | None = None):
        self.model_id = model_id
        self._api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                import anthropic
            except ImportError as e:
                raise AdapterError(
                    "the 'anthropic' package is not installed - `pip install anthropic`"
                ) from e
            if not self._api_key:
                raise AdapterError("no Anthropic API key - set ANTHROPIC_API_KEY or pass api_key=")
            self._client = anthropic.Anthropic(api_key=self._api_key)
        return self._client

    def complete(
        self, prompt: str, *, system: str | None = None, temperature: float = 0.0,
        max_tokens: int = 512, item: dict | None = None,
    ) -> AdapterResponse:
        client = self._get_client()
        import anthropic  # safe: _get_client() above already proved this import succeeds

        request = {
            "model": self.model_id,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            request["system"] = system

        start = time.monotonic()
        try:
            response = client.messages.create(**request)
        except anthropic.RateLimitError as e:
            retry_after = None
            headers = getattr(getattr(e, "response", None), "headers", None)
            if headers and headers.get("retry-after"):
                retry_after = float(headers["retry-after"])
            raise RateLimitedError(str(e), retry_after=retry_after) from e
        except anthropic.APIError as e:
            return AdapterResponse(
                text="", model_id=self.model_id, latency_s=time.monotonic() - start,
                raw_request=request, error=str(e),
            )
        latency = time.monotonic() - start

        text = "".join(
            block.text for block in response.content if getattr(block, "type", None) == "text"
        )
        return AdapterResponse(
            text=text, model_id=self.model_id, latency_s=latency,
            input_tokens=response.usage.input_tokens, output_tokens=response.usage.output_tokens,
            raw_request=request, raw_response=response.model_dump(),
        )
