"""Adapter for any locally-served OpenAI-compatible chat endpoint - vLLM's `vllm serve` and
Ollama's `/v1` API both speak this exact protocol, so one adapter class covers every model in
this run: no cloud API key, no per-provider SDK quirk, just a base_url + model name.

Distinct from adapters/openai_adapter.py (which is hardcoded to api.openai.com and OPENAI_API_KEY)
- this one always points at localhost, never needs a real key, and default-retries connection
  errors as rate limits so runner.py's backoff kicks in while a server is still warming up.
"""
from __future__ import annotations

import time

from bkp_eval.adapters.base import AdapterError, AdapterResponse, ModelAdapter, RateLimitedError


class LocalAPIAdapter(ModelAdapter):
    def __init__(
        self, model_id: str, base_url: str, served_model_name: str | None = None,
        raw_completions: bool = False,
    ):
        """model_id: label used in results logs (e.g. "sarvam-30b").
        served_model_name: the name the local server expects in the request body - defaults to
        model_id when the server was started with a matching --served-model-name / ollama tag.

        raw_completions: hits vLLM's /v1/completions (plain text-in/text-out) instead of
        /v1/chat/completions - for models with no real chat_template. 2026-08-27: discovered live
        that vLLM's chat-rendering pipeline can silently degenerate to a 1-token empty response
        for some base/instruction-tuned models even with a correct custom --chat-template
        (confirmed: navarasa-2.0-7b - a Gemma-family model, likely a <start_of_turn>/BOS-handling
        quirk in vLLM's chat renderer), while the exact same text sent as a raw completion prompt
        answers correctly. This mode builds the identical "{system}\n\n{prompt}" text
        HFTransformersAdapter's own no-template fallback uses, and sends it as a raw prompt - no
        --chat-template needed on the server at all in this mode. (Confirmed NOT a fix for every
        such model - airavata-7b's bnb-quantized checkpoint returns empty via raw completions too,
        that one's a real vLLM+bitsandbytes dequantization bug, not a chat-rendering artifact.)
        """
        self.model_id = model_id
        self.base_url = base_url
        self.served_model_name = served_model_name or model_id
        self.raw_completions = raw_completions
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                import openai
            except ImportError as e:
                raise AdapterError("the 'openai' package is not installed") from e
            self._client = openai.OpenAI(base_url=self.base_url, api_key="not-needed")
        return self._client

    def complete(
        self, prompt: str, *, system: str | None = None, temperature: float = 0.0,
        max_tokens: int = 512, item: dict | None = None,
    ) -> AdapterResponse:
        client = self._get_client()
        import openai  # safe: _get_client() above already proved this import succeeds

        if self.raw_completions:
            # Plain text-in/text-out - matches HFTransformersAdapter's own no-chat-template
            # fallback exactly, byte for byte.
            text_in = f"{system}\n\n{prompt}" if system else prompt
            request = {
                "model": self.served_model_name, "prompt": text_in,
                "temperature": temperature, "max_tokens": max_tokens,
            }
        else:
            messages = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": prompt})
            request = {
                "model": self.served_model_name, "messages": messages,
                "temperature": temperature, "max_tokens": max_tokens,
            }

        start = time.monotonic()
        try:
            if self.raw_completions:
                response = client.completions.create(**request)
            else:
                response = client.chat.completions.create(**request)
        except openai.RateLimitError as e:
            raise RateLimitedError(str(e)) from e
        except (openai.APIConnectionError, openai.APITimeoutError) as e:
            # the local server is still loading weights / not up yet - worth a backoff retry
            # rather than a hard failure logged against the model.
            raise RateLimitedError(f"local server unreachable: {e}", retry_after=5.0) from e
        except openai.APIError as e:
            return AdapterResponse(
                text="", model_id=self.model_id, latency_s=time.monotonic() - start,
                raw_request=request, error=str(e),
            )
        latency = time.monotonic() - start

        choice = response.choices[0]
        if self.raw_completions:
            text = choice.text or ""
        else:
            text = choice.message.content or ""
            if not text.strip():
                # Reasoning models (e.g. qwen3.6's OpenAI-compat shim) put the chain-of-thought in
                # an out-of-spec `reasoning` field and leave `content` empty if max_tokens is
                # exhausted before the final answer is emitted. Falling back to it beats silently
                # grading an empty string as a refusal - graders/_common.py's text extraction
                # still has to find an actual answer in there, so this is a safety net, not a
                # substitute for sizing max_tokens generously enough for the model to finish.
                extra = getattr(choice.message, "model_extra", None) or {}
                reasoning = extra.get("reasoning") or extra.get("reasoning_content")
                if reasoning:
                    text = reasoning
        usage = response.usage
        return AdapterResponse(
            text=text,
            model_id=self.model_id,
            latency_s=latency,
            input_tokens=usage.prompt_tokens if usage else None,
            output_tokens=usage.completion_tokens if usage else None,
            raw_request=request,
            raw_response=response.model_dump(),
        )
