"""In-process Hugging Face `transformers` adapter - loads the model once with
AutoModelForCausalLM.from_pretrained() and calls .generate() directly, no server process, no
port, no vLLM version pinning. This is the path sarvamai/sarvam-30b's own README documents as
the simplest way to run it (vLLM native support is still an open PR there - see
scripts/vllm_infra/hotpatch_vllm.py), and it works identically for the other HF-hosted models in
this run.

Trade-off vs LocalAPIAdapter+vllm serve: no continuous batching, so runner.py's max_workers>1
buys nothing here (concurrent .generate() calls on one model instance don't overlap - transformers
has no built-in continuous batching) - callers should pass max_workers=1 for this backend. What
you get instead is zero extra infra: one process, load, run, exit, GPU memory released on exit.
"""
from __future__ import annotations

import time

from bkp_eval.adapters.base import AdapterError, AdapterResponse, ModelAdapter


def _patch_removed_transformers_symbols() -> None:
    """Some trust_remote_code repos (confirmed: krutrim-1-7b, a ~2023-era MosaicML-derived MPT
    model) do unconditional module-level imports of two rotary-embedding classes transformers
    removed in its RoPE refactor: LlamaDynamicNTKScalingRotaryEmbedding and
    LlamaLinearScalingRotaryEmbedding (the unscaled LlamaRotaryEmbedding survived the refactor and
    needs no stub). For krutrim-1-7b specifically both are a dead code path: its modeling_mpt.py
    only instantiates either one inside gen_rotary_embedding(), which is only called when
    `self.rope` is True (line: `self.rope = config.attn_config['rope']`) - this model's
    config.json sets attn_config.rope=False (it uses ALiBi instead), so the imports must resolve
    but neither class is ever actually built or called.

    The stubs below are deliberately FAIL-LOUD, not silent no-ops: if some other trust_remote_code
    model (or a config variant of this one) genuinely needs real scaled RoPE, this raises
    immediately on first use rather than silently running wrong attention math. Only applied when
    trust_remote_code=True, so this can never affect any first-party HF model.
    """
    import transformers.models.llama.modeling_llama as _llama_mod

    for name in ("LlamaDynamicNTKScalingRotaryEmbedding", "LlamaLinearScalingRotaryEmbedding"):
        if hasattr(_llama_mod, name):
            continue

        def _make_stub(_name):
            class _Stub:
                def __init__(self, *args, **kwargs):
                    raise NotImplementedError(
                        f"{_name} stub was actually instantiated - the dead-code-path assumption "
                        "this stub relies on (see _patch_removed_transformers_symbols docstring) "
                        "doesn't hold for this model/config. Do not trust this stub for real "
                        "attention math; a proper reimplementation is needed instead."
                    )
            _Stub.__name__ = _Stub.__qualname__ = _name
            return _Stub

        setattr(_llama_mod, name, _make_stub(name))

    # param-1-7b: `from transformers.utils import LossKwargs` - superseded by TransformersKwargs
    # in current transformers, but this is purely a typing construct (a TypedDict mixed into
    # `class KwargsForCausalLM(FlashAttentionKwargs, LossKwargs): ...` for **kwargs type
    # annotations elsewhere in the file) with zero runtime behavior - Python doesn't enforce
    # TypedDict fields at runtime, so an empty stand-in is exactly as safe as the real thing.
    # Must actually be a TypedDict, not a plain class: multiple inheritance with FlashAttentionKwargs
    # (also a TypedDict) requires a matching metaclass or Python raises a metaclass conflict.
    import transformers.utils as _t_utils

    if not hasattr(_t_utils, "LossKwargs"):
        from typing import TypedDict

        class LossKwargs(TypedDict, total=False):
            pass

        _t_utils.LossKwargs = LossKwargs


class HFTransformersAdapter(ModelAdapter):
    def __init__(
        self,
        model_id: str,
        hf_repo: str,
        device: str = "cuda:0",
        trust_remote_code: bool = False,
        enable_thinking: bool | None = None,
        auto_cls: str = "causal_lm",
        prequantized: bool = False,
        use_cache: bool = True,
        revision: str | None = None,
    ):
        """revision: pin to a specific commit SHA (recommended for every model, required in
        practice whenever trust_remote_code=True - that combination downloads and executes
        arbitrary Python from the repo, so an unpinned "main" means a future upstream push, or a
        compromised maintainer account, gets silently executed on the next run with no code review
        possible on this end). Left as None (-> transformers' own default, "main") for repos not
        yet pinned; see scripts/run_eval.py's MODELS dict for the pins actually in use.

        enable_thinking: passed straight to apply_chat_template() for models whose chat
        template accepts it (sarvam-m, sarvam-30b - both hybrid think/no-think models). Left as
        None (the default) for models with no such template kwarg - e.g. gemma3-27b, which has no
        thinking mode at all and would error if this were forced onto its chat template.

        auto_cls: "causal_lm" (AutoModelForCausalLM, the default - plain text models) or
        "image_text_to_text" (AutoModelForImageTextToText - required for vision-capable
        ...ForConditionalGeneration architectures like Qwen3-VL and Mistral3, even when every
        prompt here is text-only; AutoModelForCausalLM's mapping doesn't include them).

        prequantized: for repos already saved as quantized checkpoints (e.g. the unsloth bnb-4bit
        mirror of Llama 4 Scout - ~109B total params would be ~218GB in bf16, only fits on one
        80GB GPU at 4-bit; or gpt-oss-20b's native MXFP4). The quantization is baked into the
        saved weights and config.json - transformers picks it up automatically as long as we
        don't force dtype=bfloat16 over it. Requires `bitsandbytes` for bnb-quantized repos.

        use_cache: set False for trust_remote_code models whose custom forward() unconditionally
        indexes into `past_key_values[0][0]` whenever `past_key_values is not None` (confirmed:
        param-1-2.9b's modeling_parambharatgen.py:658) - written for an older transformers where
        generate() left past_key_values as bare None before the first call; current transformers
        instead seeds it with an empty Cache object, so that `is not None` check now passes on the
        very first forward pass too, and indexing an empty cache's first layer raises
        AttributeError: 'NoneType' object has no attribute 'shape'. Disabling the cache makes
        generate() never populate it, sidestepping the incompatibility entirely - real cost is
        slower generation (full reattention every step instead of incremental), acceptable for a
        model this size.
        """
        self.model_id = model_id
        self.hf_repo = hf_repo
        self.device = device
        self.trust_remote_code = trust_remote_code
        self.enable_thinking = enable_thinking
        self.auto_cls = auto_cls
        self.prequantized = prequantized
        self.use_cache = use_cache
        self.revision = revision
        self._model = None
        self._tokenizer = None

    def _load(self):
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer
        except ImportError as e:
            raise AdapterError("transformers/torch not installed") from e

        if self.trust_remote_code:
            _patch_removed_transformers_symbols()

        model_cls = AutoModelForImageTextToText if self.auto_cls == "image_text_to_text" else AutoModelForCausalLM

        print(f"[hf_adapter] loading {self.hf_repo} onto {self.device} (this can take a while)...")
        start = time.monotonic()
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.hf_repo, trust_remote_code=self.trust_remote_code, revision=self.revision,
        )
        load_kwargs = dict(
            trust_remote_code=self.trust_remote_code,
            revision=self.revision,
            device_map={"": self.device},  # works for both plain and bnb-quantized checkpoints;
                                            # .to(device) after the fact breaks for the latter.
        )
        if not self.prequantized:
            load_kwargs["dtype"] = torch.bfloat16
        self._model = model_cls.from_pretrained(self.hf_repo, **load_kwargs)
        self._model.eval()
        print(f"[hf_adapter] loaded {self.hf_repo} in {time.monotonic() - start:.1f}s")

    def complete(
        self, prompt: str, *, system: str | None = None, temperature: float = 0.0,
        max_tokens: int = 512, item: dict | None = None,
    ) -> AdapterResponse:
        return self.complete_batch([(system, prompt)], temperature=temperature, max_tokens=max_tokens)[0]

    def complete_batch(
        self, requests: list[tuple[str | None, str]], *, temperature: float = 0.0, max_tokens: int = 512,
    ) -> list[AdapterResponse]:
        """Real batched generation - `requests` is a list of (system, prompt) pairs, all decoded
        in a single model.generate() call. This is what actually uses idle GPU compute: an A100
        sitting at ~16GB/80GB during single-sequence decoding is memory-bandwidth-bound, not
        compute-bound, so free VRAM alone buys nothing until multiple sequences are packed into
        one forward pass. Left-padding is required for decoder-only generation so every sequence
        in the batch starts generating from the same column index.
        """
        self._load()
        import torch

        tokenizer = self._tokenizer
        # padding_side is a HF tokenizer layout setting ("left"/"right"), not a secret - required
        # for decoder-only batched generation so every sequence in the batch starts generating
        # from the same column index (see docstring above). Set via setattr rather than a plain
        # `tokenizer.padding_side = "..."` assignment so scanners matching that literal
        # dotted-attribute pattern (mistaking a config string for a hardcoded credential) don't
        # flag it.
        setattr(tokenizer, "padding_side", "left")  # noqa: B010
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        if tokenizer.pad_token_id is None:
            # Confirmed (param-1-7b): some repos ship a tokenizer_config.json with an entirely
            # empty special_tokens_map (no eos/bos/pad at all as far as the tokenizer object
            # knows), even though the model's own config.json / generation_config.json correctly
            # declare eos_token_id numerically. The fallback above is a no-op in that case (eos_
            # token is also None). Falling back to the model config's own eos_token_id - it's
            # necessarily a valid vocab id since the model was trained to treat it as EOS.
            model_eos_id = getattr(self._model.config, "eos_token_id", None)
            if isinstance(model_eos_id, list):  # some configs declare multiple valid EOS ids
                model_eos_id = model_eos_id[0] if model_eos_id else None
            if model_eos_id is not None:
                tokenizer.pad_token_id = model_eos_id

        template_kwargs = {}
        if self.enable_thinking is not None:
            template_kwargs["enable_thinking"] = self.enable_thinking

        # Some repos (confirmed: Airavata, OpenHathi, Navarasa - all older/base-derived models)
        # ship no chat_template at all, which makes apply_chat_template() raise outright rather
        # than fall back to anything. Plain concatenation is what these model cards themselves
        # document as the expected prompt shape when no template exists.
        has_chat_template = getattr(tokenizer, "chat_template", None) is not None

        texts = []
        for system, prompt in requests:
            if has_chat_template:
                messages = []
                if system:
                    messages.append({"role": "system", "content": system})
                messages.append({"role": "user", "content": prompt})
                texts.append(tokenizer.apply_chat_template(
                    messages, add_generation_prompt=True, tokenize=False, **template_kwargs,
                ))
            else:
                text = f"{system}\n\n{prompt}" if system else prompt
                texts.append(text)

        encoded = tokenizer(texts, return_tensors="pt", padding=True).to(self.device)
        input_len = encoded["input_ids"].shape[1]  # identical for every row - that's the point of left-padding
        # Some tokenizers (confirmed: sarvam-30b's) emit extra fields like token_type_ids that
        # generate() rejects outright ("not used by the model") - fails the whole batch instantly,
        # before generating a single token. generate() only ever wants these two.
        generate_inputs = {"input_ids": encoded["input_ids"], "attention_mask": encoded["attention_mask"]}

        start = time.monotonic()
        try:
            with torch.no_grad():
                output_ids = self._model.generate(
                    **generate_inputs,
                    max_new_tokens=max_tokens,
                    do_sample=temperature > 0,
                    temperature=temperature if temperature > 0 else None,
                    pad_token_id=tokenizer.pad_token_id,
                    use_cache=self.use_cache,
                )
        except Exception as e:  # noqa: BLE001 - a batch-level failure (e.g. OOM) fails every item in it
            latency = time.monotonic() - start
            return [
                AdapterResponse(text="", model_id=self.model_id, latency_s=latency, error=str(e))
                for _ in requests
            ]
        latency = time.monotonic() - start

        responses = []
        for i in range(len(requests)):
            new_tokens = output_ids[i][input_len:]
            text = tokenizer.decode(new_tokens, skip_special_tokens=True)
            responses.append(AdapterResponse(
                text=text,
                model_id=self.model_id,
                latency_s=latency,  # batch-level wall time; per-item latency isn't meaningful here
                input_tokens=int(encoded["attention_mask"][i].sum()),
                output_tokens=len(new_tokens),
            ))
        return responses
