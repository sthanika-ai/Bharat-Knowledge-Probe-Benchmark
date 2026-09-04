#!/usr/bin/env python3
"""CLI entrypoint for the eval harness: wires load_corpus() + a ModelAdapter + runner.run()
together from the command line.

Every model here is served locally (Ollama or `vllm serve`) via LocalAPIAdapter - no cloud API
key touches this script. Two modes:

  smoke   - first N items (default 50), R1+R2, n=1 sample: proves the adapter/prompt/grading
            pipeline works end-to-end against a real server before spending hours on the full run.
  full    - every item + every control (so Locale Gap Delta is computable), R1+R2, n=3 samples.

Usage:
  python scripts/run_eval.py --model sarvam-m --mode smoke
  python scripts/run_eval.py --model sarvam-m --mode full
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from bkp_eval.adapters.hf_adapter import HFTransformersAdapter  # noqa: E402
from bkp_eval.adapters.local_api_adapter import LocalAPIAdapter  # noqa: E402
from bkp_eval.items import load_corpus  # noqa: E402
from bkp_eval.runner import RunConfig, run  # noqa: E402
from bkp_eval.score import score, _format_report  # noqa: E402

# One entry per model in this run. "ollama" backend hits Ollama's built-in OpenAI-compatible
# endpoint (qwen3.6's hybrid attention+SSM+vision architecture isn't supported by vLLM yet).
# "hf" backend loads the model in-process via transformers .generate() - no server, no vLLM
# version pinning; this is what sarvamai/sarvam-30b's own README documents as the simplest path
# (native vLLM support there is still an open PR). All four HF-hosted models already have their
# weights cached locally.
MODELS = {
    "qwen3.6-27b": dict(
        base_url="http://localhost:11434/v1",
        served_model_name="qwen3.6:27b",
        backend="ollama",
        # This is a reasoning model - its chain-of-thought alone runs 500+ tokens on anything
        # non-trivial (confirmed empirically: a 512-token budget was exhausted mid-reasoning with
        # zero answer content emitted). Needs real headroom to ever reach an answer.
        max_tokens=4096,
        # Matches the ollama service's OLLAMA_NUM_PARALLEL=4 (systemd override.conf) - going
        # higher just queues client-side with no server-side benefit.
        max_workers=4,
    ),
    "llama3.1-8b-instruct": dict(
        backend="hf",
        hf_repo="NousResearch/Meta-Llama-3.1-8B-Instruct",
        # Smallest model here (~16.5GB weights) with the most free VRAM headroom of any model in
        # this run - can afford a bigger batch than gemma3-27b/sarvam-m/sarvam-30b.
        batch_size=16,
    ),
    "gemma3-27b": dict(
        backend="hf",
        hf_repo="unsloth/gemma-3-27b-it",
        device="cuda:1",
        # ~54GB bf16 weights, so budget for KV cache is smaller than llama's - 8 real
        # simultaneously-decoded sequences per generate() call, not 8 queued threads.
        batch_size=8,
        # No thinking mode exists for gemma3 (verified: no enable_thinking in its chat template,
        # no mention in its model card) - runs as a plain instruct model.
    ),
    "sarvam-m": dict(
        # Plain MistralForCausalLM - no custom code, natively supported by vLLM. Switched from
        # the "hf" backend after confirming HF's sequential batch_size=1 path (forced by thinking
        # mode's batch-penalty problem, see sarvam-30b below) couldn't finish even a 30-item smoke
        # within an hour. vLLM's continuous batching has no such penalty - each sequence finishes
        # independently, so thinking-mode's variable output length costs nothing extra.
        backend="vllm", hf_repo="sarvamai/sarvam-m",
        base_url="http://localhost:8103/v1", served_model_name="sarvam-m",
        # 2026-08-27: measured Running=16/Waiting=0 with only ~50-57% GPU KV cache used at
        # max_workers=16 - headroom to run more concurrent sequences without hitting cache
        # pressure. Bumped concurrency and gpu_util together to use it.
        vllm_port=8103, vllm_gpu_util=0.8, vllm_max_len=8192,
        max_tokens=4096, max_workers=32,
        # Pinned for reproducibility - confirmed this is exactly
        # what "main" already resolved to when first downloaded (2026-08-27), so pinning here is
        # documentation, not a behavior change. NOT the same generation config as this repo's own
        # MILU-benchmark investigation (max_new_tokens=1536) - that's a different, easier task;
        # our harder items empirically needed the full 4096 to avoid truncating mid-reasoning.
        revision="01534a53c46f2788e392dbb3d994e0fa8f04d3fd",
    ),
    "sarvam-30b": dict(
        # 2026-08-27: switched from "hf" to vLLM after applying sarvamai/sarvam-30b's own
        # hotpatch_vllm.py (scripts/vllm_infra/) - native support is still an open vLLM PR, the
        # hotpatch adds SarvamMoEForCausalLM/SarvamMLAForCausalLM to registry.py and installs the
        # repo's own custom modeling code. Verified: ModelRegistry recognizes both classes after
        # patching. Same payoff as sarvam-m's vllm switch - HF's sequential batch_size=1 path
        # (forced by thinking mode's batch-penalty problem) was too slow even for a 30-item smoke.
        backend="vllm", hf_repo="sarvamai/sarvam-30b",
        base_url="http://localhost:8115/v1", served_model_name="sarvam-30b",
        vllm_port=8115, vllm_gpu_util=0.85, vllm_max_len=8192,
        vllm_extra_args=["--trust-remote-code"],  # custom arch needs the repo's own modeling code
        # 2026-08-27: calibrated via smoke/live testing: 32->64->128 scaled cleanly (899.7 ->
        # 1829.3 -> 4018.8 tok/s, KV cache 3.4% -> 7.3% -> 16.4%). 256 looked promising at first
        # (~5100 tok/s) but KV cache climbed to 100% at steady state, causing Running to drop and
        # Waiting>0 to appear - real backpressure, unsafe for a long unattended run. Settled on
        # 128 as the last value with clean headroom (Waiting=0, cache well under any risk zone).
        max_tokens=4096, max_workers=128,
    ),
    # --- Extended roster (2026-08-26 request) - GPU0 is occupied by the user's own separate
    # lm_eval job right now, so everything below targets cuda:1 too. All repo ids/architectures/
    # gating verified live against the HF API before being added here.
    "sarvam-1": dict(
        backend="hf", hf_repo="sarvamai/sarvam-1", device="cuda:1", batch_size=32,
    ),
    "krutrim-1-7b": dict(
        # 2026-08-27: REVERTED from vllm - the comment this replaced claimed vLLM's own native
        # MptForCausalLM handles grouped_query_attention correctly, but live testing just proved
        # that wrong: vLLM's ModelConfig validation rejects attn_type=grouped_query_attention
        # outright ("has to be either multihead_attention or multiquery_attention"), a hard
        # pydantic validation error before the server can even start - not a version fluke, it's
        # this vLLM install's actual MPT support. Back to the HF path (trust_remote_code=True
        # bypasses transformers' own older MptConfig limitation, per the original fix for this
        # model) - no continuous batching there, so max_workers is irrelevant (HFTransformersAdapter
        # ignores it); batch_size is the real lever, set conservatively since it's unquantized 7B.
        # 2026-08-27: BLOCKED on the HF path too - this custom modeling code (MosaicML-derived
        # MPT, ~2023-era) hits two more incompatibilities with the installed transformers 4.57.6:
        # (1) its flash_attn_triton.py needs `triton_pre_mlir`, not on PyPI - worked around with a
        # stub package (scripts/vllm_infra has no copy, it's an empty triton_pre_mlir/ dropped
        # straight into .venv-vllm/site-packages; safe because that file is only ever reached when
        # attn_config.attn_impl=="triton", and this config sets "torch") - genuinely a dead code
        # path, so a no-op stub is correct there.
        # (2) modeling_mpt.py does `from transformers.models.llama.modeling_llama import
        # LlamaDynamicNTKScalingRotaryEmbedding` at module top level (unconditional, actually used
        # at model init) - that class no longer exists after transformers' rotary-embedding
        # refactor. Unlike (1) this is NOT a dead branch - patching around a missing rotary-
        # embedding class risks silently wrong attention math, not just an import error. Needs
        # either an older pinned transformers (in its own venv - too risky to downgrade the
        # shared .venv-vllm every other model here depends on) or the repo owner's own fix.
        # Leaving this one blocked rather than guessing at a replacement implementation.
        # 2026-08-27: max_tokens capped 512->128 - this tokenizer has no chat_template, so
        # hf_adapter's plain-text fallback prompts it in raw-completion style, which this
        # instruct-tuned model apparently never learned to naturally terminate from (every single
        # smoke/full response hit exactly 512/512 output tokens, no early EOS - confirmed via
        # smoke.jsonl). At max_tokens=512 the full run projected ~36.8h from observed batch
        # latency (244.8-380.5s/batch of 16). The actual answer consistently appears in the first
        # ~20-50 tokens before it rambles on to unrelated topics (verified across multiple smoke
        # samples) - capping bounds the wasted-generation cost without losing the extractable
        # answer; doesn't fix the rambling itself, just stops paying for it.
        backend="hf", hf_repo="Krutrim-AI-Labs/krutrim-1-instruct", device="cuda:0",
        trust_remote_code=True, batch_size=16, max_tokens=128,
        # Pinned 2026-08-28 (security review): trust_remote_code=True executes this repo's own
        # Python, so an unpinned "main" would silently re-execute whatever the maintainer pushes
        # next. This is what "main" resolved to as of pinning - same rationale as sarvam-m's pin.
        revision="ef1e55353589e1b53f3c79b9526666ac8901df11",
    ),
    "krutrim-2-12b": dict(
        backend="hf", hf_repo="Krutrim-AI-Labs/Krutrim-2-instruct", device="cuda:1", batch_size=8,
    ),
    "param-1-2.9b": dict(
        # 2026-08-27: the earlier "100% refusal" label was wrong - every row was actually crashing
        # with AttributeError: 'NoneType' object has no attribute 'shape' (modeling_
        # parambharatgen.py:658 does past_key_values[0][0].shape unconditionally whenever
        # `past_key_values is not None` - written for older transformers where generate() left it
        # as bare None pre-first-call; current transformers seeds an empty Cache object instead,
        # so that check now passes immediately and indexing its empty first layer raises).
        # Tried patching the cached modeling file to guard that line (and a second, matching
        # unsafe access at the per-layer attention level) to restore real KV caching - but this
        # produced a WORSE failure: a raw CUDA device-side assert (index out of bounds), almost
        # certainly from the two patched guards disagreeing about cache state and producing an
        # inconsistent position/sequence-length calculation somewhere downstream. That's not just
        # a crash risk, it's a risk of silently WRONG generation in cases that don't crash -
        # unacceptable for a benchmark eval. Reverted the modeling-file patch entirely; use_cache=
        # False is the settled choice - slower (no incremental KV cache, recomputes attention over
        # the whole prefix every step) but proven correct (verified real, sensible answers).
        # Lowering batch_size alone didn't help (4 was still stuck >3.5min into a 60-row smoke,
        # ~76GB used) - the real cost driver here is max_tokens, not batch width: no-cache
        # generation is O(max_tokens^2) per sequence regardless of batch size. Real answers
        # complete in 35-173 tokens on a quick manual test, well under the 512 default, so capping
        # max_tokens (same lever, same reasoning as krutrim-1-7b's fix) should cut the dominant
        # cost by roughly (128/512)^2 =~ 16x rather than the ~2x batch_size alone was buying.
        backend="hf", hf_repo="bharatgenai/Param-1-2.9B-Instruct", device="cuda:1", batch_size=8,
        max_tokens=128, trust_remote_code=True, use_cache=False,  # custom ParamBharatGenForCausalLM arch
        # Pinned 2026-08-28 (security review) - see krutrim-1-7b's pin comment for the rationale.
        revision="f9b32a727458727ff515fea5d82df00ae9b7ad56",
    ),
    "param-1-7b": dict(
        # 2026-08-27: was blocked on `ImportError: cannot import name 'LossKwargs'` -
        # transformers renamed/removed this (superseded by TransformersKwargs); it's purely a
        # typing construct (TypedDict mixin for a **kwargs annotation), zero runtime behavior, so
        # hf_adapter's _patch_removed_transformers_symbols() stubs it in when trust_remote_code.
        # Second bug found once that cleared: tokenizer_config.json ships an entirely empty
        # special_tokens_map (no bos/eos/pad at all) even though config.json/generation_config.json
        # correctly declare eos_token_id=2 numerically - hf_adapter's pad_token fallback now falls
        # further back to the model config's own eos_token_id when the tokenizer's own is None too.
        # Repo has no "-Instruct" suffix (unlike the 2.9B sibling) - confirmed base/pretrained, not
        # instruction-tuned: a manual test just repeated the input question back verbatim rather
        # than answering, classic base-model behavior on an isolated question with no
        # chat_template. Expect a low score here, same category as openhathi-7b/sarvam-1 - that's
        # real model capability, not a bug to chase further. Confirmed via the real smoke test:
        # all 16 rows of the first batch hit exactly 512/512 output tokens generating pure
        # newlines - genuinely degenerate, not stopping ever. max_tokens capped 512->128 first,
        # still not enough: the full run at 128 was STILL hitting the cap on every single row
        # (422.6s/batch of 16, projecting ~36.8h) - same non-stopping behavior just at a lower
        # ceiling. Since the content is confirmed uniformly degenerate garbage regardless of
        # length, there's no real signal being cut short by capping further. Dropped to 32
        # (~4x less generation per row than 128) purely for wall-clock; first 144 rows already on
        # disk from the max_tokens=128 attempt stay as-is (resumability doesn't distinguish by
        # max_tokens, and both settings produce the same non-answer either way).
        backend="hf", hf_repo="bharatgenai/Param-1-7B", device="cuda:0", batch_size=16,
        max_tokens=32, trust_remote_code=True,  # custom Param1MoEForCausalLM arch
        # Pinned 2026-08-28 (security review) - see krutrim-1-7b's pin comment for the rationale.
        revision="e50c644204ae369594cd7ad84305eb505a1da1c3",
    ),
    "airavata-7b": dict(
        # ai4bharat/Airavata itself is gated (401 without a token) - this ungated bnb-8bit mirror
        # has identical base weights, just quantized (smaller footprint as a bonus).
        # 2026-08-27: REVERTED to hf (3rd flip, settling here) - the missing-chat_template 400 was
        # real and --chat-template did fix IT, but exposed a deeper problem: vLLM's bitsandbytes
        # loader produces degenerate output (1 token, decodes to "") on many prompts that HF's own
        # bnb loading path answers correctly - confirmed via direct /v1/completions with the exact
        # same raw text HFTransformersAdapter sends (bypassing chat template entirely, still
        # empty), and confirmed HF's identical text DOES get real answers (this model's own full
        # HF run logged 45% refusal, not 100% - most prompts work fine there). This looks like a
        # genuine dequantization/precision discrepancy between vLLM's bnb loader and transformers'
        # for this specific checkpoint, not a template or prompt-format issue - not something to
        # chase further here. HF backend is the only currently-reliable path for this model;
        # prequantized=True since the mirror repo's weights are already saved as bnb-8bit.
        backend="hf", hf_repo="maitreyaz/Airavata-8bit", device="cuda:0",
        # calibrated: batch_size 16->96 (6x, 4.4x more GPU mem: 16GB->72GB) only bought 0.72->0.85
        # rows/s (~18%) - HF's lock-step generate() means the whole batch waits on the slowest
        # sequence each step, so bigger batches don't scale the way vLLM's continuous batching
        # does. Settled on 32: most of the achievable gain without batch=96's near-OOM margin.
        prequantized=True, batch_size=32,
    ),
    "openhathi-7b": dict(
        # 2026-08-27: back on vllm (2nd revert) - same fix as airavata-7b: the real blocker was
        # the missing chat_template, not max_model_len (that fix - 4096 not 8192 - is still
        # correct and kept). --chat-template restores vLLM's continuous batching. Unlike
        # airavata-7b this repo isn't bnb-quantized, so it doesn't hit that loader bug - verified
        # real (non-degenerate) generation via direct curl before calibrating.
        # Calibrated via smoke ramp (--limit 200): 32->64->128->256 all clean (KV cache stayed
        # under 22%, Waiting=0 throughout). Diminishing returns past 128 (29.3s->19.7s->16.8s->
        # 14.5s) - settled on 128, same reasoning as gemma3-12b-int4.
        backend="vllm", hf_repo="sarvamai/OpenHathi-7B-Hi-v0.1-Base",
        base_url="http://localhost:8106/v1", served_model_name="openhathi-7b",
        vllm_port=8106, vllm_gpu_util=0.85, vllm_max_len=4096, max_workers=128,
        vllm_extra_args=[
            "--chat-template", str(REPO_ROOT / "scripts" / "vllm_infra" / "plain_concat_template.jinja"),
        ],
    ),
    "navarasa-2.0-7b": dict(
        # 2026-08-27: back on vllm (2nd revert) - the earlier "SMOKE FAILED ... 100% refusal" was
        # never a genuine model refusal, it was every row 400ing on the missing chat_template
        # (mis-bucketed as "refusal" by the grader). --chat-template fixed the 400, but exposed a
        # second bug: vLLM's chat-rendering pipeline returns an empty 1-token response for this
        # Gemma-family model even with a correct template (verified: same text sent raw via
        # /v1/completions instead of /v1/chat/completions answers correctly - "### Input:...
        # ### Response:\n{"value": 500000}", exactly right). Switched to raw_completions=True
        # (see LocalAPIAdapter) - no --chat-template needed at all in this mode, it bypasses
        # vLLM's chat renderer entirely and sends the same plain text HF's fallback would.
        # Calibrated via smoke ramp (--limit 200): 32/64/128 all flat (13.0s/12.1s/12.5s) - this
        # model's generations are short enough that client-side overhead dominates over server
        # capacity well before 32. Settled on 64 (no measured benefit above it, some safety margin
        # below the point where it stopped mattering).
        backend="vllm", hf_repo="Telugu-LLM-Labs/Indic-gemma-7b-finetuned-sft-Navarasa-2.0",
        base_url="http://localhost:8107/v1", served_model_name="navarasa-2.0-7b",
        vllm_port=8107, vllm_gpu_util=0.85, vllm_max_len=8192, max_workers=64,
        raw_completions=True,
    ),
    "gemma3-12b": dict(
        # google/gemma-3-12b-it is gated - same ungated-mirror policy as gemma3-27b/gemma3-4b.
        backend="hf", hf_repo="unsloth/gemma-3-12b-it", device="cuda:1", batch_size=8,
    ),
    "gemma3-12b-int4": dict(
        # Ungated mirror of google/gemma-3-12b-it-qat-int4-unquantized (that exact repo is gated).
        # 2026-08-27: calibrated via smoke ramp (--limit 200): 32->64->128->256 all clean (KV
        # cache usage stayed under 5%, Waiting=0 throughout - this quantized 12B has plenty of
        # headroom). Wall time kept improving but with sharply diminishing returns past 128
        # (49.9s -> 36.1s -> 30.0s -> 26.8s for 32/64/128/256) - the last doubling only bought
        # ~11%, likely client-side overhead rather than server capacity by that point. Settled on
        # 128: captures nearly all the available throughput without pushing concurrency past the
        # point of real measured benefit.
        backend="vllm", hf_repo="unsloth/gemma-3-12b-it-qat-int4-bnb-4bit",
        base_url="http://localhost:8108/v1", served_model_name="gemma3-12b-int4",
        vllm_port=8108, vllm_gpu_util=0.85, vllm_max_len=8192, max_workers=128,
        vllm_extra_args=["--quantization", "bitsandbytes"],
    ),
    "gemma3-4b": dict(
        backend="hf", hf_repo="unsloth/gemma-3-4b-it", device="cuda:1", batch_size=48,
    ),
    "gpt-oss-20b": dict(
        # Native MXFP4 - vLLM auto-detects the quantization from config.json, no extra flag needed.
        backend="vllm", hf_repo="openai/gpt-oss-20b",
        base_url="http://localhost:8109/v1", served_model_name="gpt-oss-20b",
        vllm_port=8109, vllm_gpu_util=0.85, vllm_max_len=8192, max_workers=16,
        max_tokens=2048,  # reasoning model - give it real headroom, same lesson as qwen3.6/sarvam
    ),
    "qwen3-vl-8b": dict(
        backend="vllm", hf_repo="Qwen/Qwen3-VL-8B-Instruct",
        base_url="http://localhost:8110/v1", served_model_name="qwen3-vl-8b",
        vllm_port=8110, vllm_gpu_util=0.85, vllm_max_len=8192, max_workers=16,
    ),
    "qwen2.5-7b": dict(
        backend="vllm", hf_repo="Qwen/Qwen2.5-7B-Instruct",
        base_url="http://localhost:8111/v1", served_model_name="qwen2.5-7b",
        vllm_port=8111, vllm_gpu_util=0.85, vllm_max_len=8192, max_workers=16,
    ),
    "mistral-small-3.1-24b": dict(
        backend="vllm", hf_repo="mistralai/Mistral-Small-3.1-24B-Instruct-2503",
        base_url="http://localhost:8112/v1", served_model_name="mistral-small-3.1-24b",
        vllm_port=8112, vllm_gpu_util=0.85, vllm_max_len=8192, max_workers=16,
    ),
    "llama4-scout": dict(
        # meta-llama/Llama-4-Scout-17B-16E-Instruct is gated AND ~109B total params (~218GB bf16 -
        # doesn't fit at all). This ungated unsloth bnb-4bit mirror is ~55GB, fits on one 80GB GPU.
        backend="vllm", hf_repo="unsloth/Llama-4-Scout-17B-16E-Instruct-unsloth-bnb-4bit",
        base_url="http://localhost:8113/v1", served_model_name="llama4-scout",
        vllm_port=8113, vllm_gpu_util=0.85, vllm_max_len=8192, max_workers=8,
        vllm_extra_args=["--quantization", "bitsandbytes"],
    ),
    "phi-4-14b": dict(
        backend="vllm", hf_repo="microsoft/phi-4",
        base_url="http://localhost:8114/v1", served_model_name="phi-4-14b",
        vllm_port=8114, vllm_gpu_util=0.85, vllm_max_len=8192, max_workers=16,
    ),
    # NOT included yet: "Gemma 4 12B" (google/gemma-4-12b-it, confirmed to exist and ungated) -
    # its Gemma4UnifiedForConditionalGeneration architecture needs transformers>=5.x; our venv is
    # pinned to 4.57.6 (matches vllm 0.15.0's tested range). Needs a separate venv before it can
    # run - flagged, not silently skipped.
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, choices=sorted(MODELS))
    parser.add_argument("--mode", required=True, choices=["smoke", "full"])
    parser.add_argument("--limit", type=int, default=None, help="override item count (smoke default: 30)")
    parser.add_argument("--n-samples", type=int, default=None, help="override sample count")
    parser.add_argument(
        "--regimes", nargs="+", default=["R1", "R2"], choices=["R1", "R2"],
        help="prompt regimes to run (default: both)",
    )
    parser.add_argument("--max-tokens", type=int, default=None, help="override; default is per-model (512 unless the model config sets its own)")
    parser.add_argument("--max-workers", type=int, default=None, help="override; default is per-model (1 unless the model config sets its own)")
    parser.add_argument("--batch-size", type=int, default=None, help="override; real batched generate() for hf backend - default is per-model (1 unless the model config sets its own)")
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "results")
    parser.add_argument("--device", type=str, default=None, help="override cfg's device (hf backend only) - e.g. run the same model config on cuda:0 instead of cuda:1")
    args = parser.parse_args()

    cfg = MODELS[args.model]
    if cfg["backend"] == "hf":
        adapter = HFTransformersAdapter(
            model_id=args.model, hf_repo=cfg["hf_repo"],
            device=args.device if args.device is not None else cfg.get("device", "cuda:0"),
            trust_remote_code=cfg.get("trust_remote_code", False),
            enable_thinking=cfg.get("enable_thinking"),
            auto_cls=cfg.get("auto_cls", "causal_lm"),
            prequantized=cfg.get("prequantized", False),
            use_cache=cfg.get("use_cache", True),
            revision=cfg.get("revision"),
        )
    else:
        adapter = LocalAPIAdapter(
            model_id=args.model, base_url=cfg["base_url"], served_model_name=cfg["served_model_name"],
            raw_completions=cfg.get("raw_completions", False),
        )

    corpus = load_corpus()
    if args.mode == "smoke":
        limit = args.limit if args.limit is not None else 30
        item_ids = sorted(corpus.items)[:limit]
        n_samples = args.n_samples if args.n_samples is not None else 1
    else:
        item_ids = sorted(corpus.items) + sorted(corpus.controls)
        if args.limit is not None:
            item_ids = item_ids[: args.limit]
        n_samples = args.n_samples if args.n_samples is not None else 3

    max_tokens = args.max_tokens if args.max_tokens is not None else cfg.get("max_tokens", 512)
    max_workers = args.max_workers if args.max_workers is not None else cfg.get("max_workers", 1)
    batch_size = args.batch_size if args.batch_size is not None else cfg.get("batch_size", 1)
    run_config = RunConfig(regimes=tuple(args.regimes), n_samples=n_samples, max_tokens=max_tokens)

    out_path = args.out_dir / args.model / f"{args.mode}.jsonl"
    print(
        f"[run_eval] model={args.model} mode={args.mode} items={len(item_ids)} "
        f"regimes={run_config.regimes} n_samples={run_config.n_samples} "
        f"max_workers={max_workers} batch_size={batch_size} -> {out_path}"
    )
    stats = run(corpus, adapter, item_ids, out_path, config=run_config, max_workers=max_workers, batch_size=batch_size)
    print(
        f"[run_eval] done: calls_made={stats.calls_made} skipped_resumed={stats.calls_skipped_resumed} "
        f"errors={stats.errors} rate_limit_retries={stats.rate_limit_retries}"
    )
    if stats.errors:
        print(f"[run_eval] WARNING: {stats.errors} calls ended in error - see {out_path} for detail")

    reports = score(out_path, corpus=corpus)
    report_path = args.out_dir / args.model / f"{args.mode}_report.txt"
    with open(report_path, "w", encoding="utf-8") as fh:
        for model, report in reports.items():
            text = _format_report(model, report)
            print(text)
            fh.write(text + "\n\n")
    print(f"[run_eval] report written to {report_path}")
    return 1 if stats.errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
