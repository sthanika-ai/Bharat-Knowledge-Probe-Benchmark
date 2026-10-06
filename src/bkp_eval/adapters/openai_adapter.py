"""OpenAI provider adapter (Chat Completions API).

The `openai` package happens to be installed in this environment (for unrelated reasons - no
project dependency on it was declared), but no OPENAI_API_KEY is configured, so this has never
been exercised against a live model. Same deferred-import shape as anthropic_adapter.py. Treat
as implemented-but-unverified until it's run once against a real key.
"""
from __future__ import annotations

import os
import time

from bkp_eval.adapters.base import AdapterError, AdapterResponse, ModelAdapter, RateLimitedError


class OpenAIAdapter(ModelAdapter):
    def __init__(self, model_id: str, api_key: str | None = None):
        self.model_id = model_id
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                import openai
            except ImportError as e:
                raise AdapterError("the 'openai' package is not installed") from e
            if not self._api_key:
                raise AdapterError("no OpenAI API key - set OPENAI_API_KEY or pass api_key=")
            self._client = openai.OpenAI(api_key=self._api_key)
        return self._client

    def complete(
        self, prompt: str, *, system: str | None = None, temperature: float = 0.0,
        max_tokens: int = 512, item: dict | None = None,
    ) -> AdapterResponse:
        client = self._get_client()
        import openai  # safe: _get_client() above already proved this import succeeds

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        request = {
            "model": self.model_id, "messages": messages,
            "temperature": temperature, "max_tokens": max_tokens,
        }

        start = time.monotonic()
        try:
            response = client.chat.completions.create(**request)
        except openai.RateLimitError as e:
            raise RateLimitedError(str(e)) from e
        except openai.APIError as e:
            return AdapterResponse(
                text="", model_id=self.model_id, latency_s=time.monotonic() - start,
                raw_request=request, error=str(e),
            )
        latency = time.monotonic() - start

        text = response.choices[0].message.content or ""
        usage = response.usage
        return AdapterResponse(
            text=text, model_id=self.model_id, latency_s=latency,
            input_tokens=usage.prompt_tokens if usage else None,
            output_tokens=usage.completion_tokens if usage else None,
            raw_request=request, raw_response=response.model_dump(),
        )
