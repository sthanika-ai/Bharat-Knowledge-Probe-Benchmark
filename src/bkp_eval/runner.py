"""runner.py - the resumable, rate-limit-aware evaluation loop.

Calls a model adapter over every (item, regime, sample) triple and appends each raw response to
a results log that score.py reads directly (same JSONL shape score.py's `load_results` expects,
plus a few extra logging fields it simply ignores).

Resumability: before calling the adapter, checks whether (item_id, regime, sample_index) is
already present in the output file from a prior run and skips it - an interrupted run (crash,
Ctrl-C, exhausted retries) costs at most the one in-flight call, never a restart. Every completed
call (success OR final failure) is written and flushed immediately, never buffered to the end.

Note: a row logged with a non-null "error" counts as "done" for resume purposes - re-running
`run()` will NOT automatically retry a call that already gave up after max_retries. To force a
retry, remove that line from the results file first (or filter it out when writing to a new
target path) - that's a deliberate choice so a persistently-broken item doesn't burn retries
forever every time the run is resumed.
"""
from __future__ import annotations

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from bkp_eval.adapters.base import AdapterError, AdapterResponse, ModelAdapter, RateLimitedError
from bkp_eval.items import Corpus
from bkp_eval.prompts import build_prompt


@dataclass
class RunConfig:
    regimes: tuple[str, ...] = ("R1", "R2")
    n_samples: int = 3
    temperature: float = 0.0
    max_tokens: int = 512
    max_retries: int = 5
    backoff_base_s: float = 1.0


@dataclass
class RunStats:
    calls_made: int = 0
    calls_skipped_resumed: int = 0
    errors: int = 0
    rate_limit_retries: int = 0


def _already_done(out_path: Path) -> set[tuple[str, str, int]]:
    done: set[tuple[str, str, int]] = set()
    if not out_path.exists():
        return done
    with open(out_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            done.add((row["item_id"], row["regime"], row.get("sample_index", 0)))
    return done


def _call_with_retry(adapter, prompt, system, config, item, stats, sleep_fn) -> AdapterResponse:
    attempt = 0
    while True:
        try:
            return adapter.complete(
                prompt, system=system, temperature=config.temperature,
                max_tokens=config.max_tokens, item=item,
            )
        except RateLimitedError as e:
            attempt += 1
            if attempt > config.max_retries:
                return AdapterResponse(
                    text="", model_id=adapter.model_id, latency_s=0.0,
                    error=f"gave up after {config.max_retries} rate-limit retries: {e}",
                )
            stats.rate_limit_retries += 1
            delay = e.retry_after if e.retry_after is not None else config.backoff_base_s * (2 ** (attempt - 1))
            sleep_fn(delay)
        except AdapterError as e:
            return AdapterResponse(text="", model_id=adapter.model_id, latency_s=0.0, error=str(e))


def _iter_pending(corpus, item_ids, config, done):
    """Yields (item_id, regime, sample_index, item, system, user_prompt) for every call not
    already in `done` - the shared work list both the sequential and threaded paths consume.
    """
    for item_id in item_ids:
        item = corpus.all_by_id.get(item_id)
        if item is None:
            raise KeyError(f"unknown item_id {item_id!r}")
        for regime in config.regimes:
            system, user_prompt = build_prompt(item, regime)
            for sample_index in range(config.n_samples):
                key = (item_id, regime, sample_index)
                if key in done:
                    continue
                yield item_id, regime, sample_index, item, system, user_prompt


def _row_for(item_id, regime, sample_index, model_id, response):
    return {
        "item_id": item_id, "model": model_id, "regime": regime,
        "sample_index": sample_index, "response": response.text,
        "latency_s": response.latency_s, "input_tokens": response.input_tokens,
        "output_tokens": response.output_tokens, "error": response.error,
    }


def run(
    corpus: Corpus,
    adapter: ModelAdapter,
    item_ids: list[str],
    out_path: Path | str,
    config: RunConfig | None = None,
    sleep_fn=time.sleep,
    max_workers: int = 1,
    batch_size: int = 1,
) -> RunStats:
    """max_workers > 1 fans calls out across a thread pool - safe because each call is an
    independent HTTP request (releases the GIL while waiting) and every shared mutation (the
    output file, `stats`, `done`) goes through `lock`. Resumability, retry/backoff, and the
    on-disk row shape are identical to the sequential path (max_workers=1, the default) - only
    the order rows land in the file changes, which score.py never depends on.

    batch_size > 1 takes a different path entirely: it requires `adapter.complete_batch()` (only
    HFTransformersAdapter has this) and packs that many (item, regime, sample) prompts into one
    real batched generate() call - the only way to actually use idle GPU compute during
    in-process decoding, since threads calling .generate() on the same model don't parallelize
    (Python-level CUDA calls serialize). Mutually exclusive with max_workers: batching already
    keeps the GPU busy, and there is no retry-with-backoff here - local inference doesn't rate
    limit, so a batch-level exception (e.g. OOM) is logged as an error for every item in it rather
    than retried.
    """
    config = config or RunConfig()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = _already_done(out_path)
    stats = RunStats()
    lock = threading.Lock()

    pending = list(_iter_pending(corpus, item_ids, config, done))
    with lock:
        stats.calls_skipped_resumed += sum(
            1 for item_id in item_ids for regime in config.regimes for s in range(config.n_samples)
        ) - len(pending)

    with open(out_path, "a", encoding="utf-8") as out_fh:

        def handle_one(work):
            item_id, regime, sample_index, item, system, user_prompt = work
            response = _call_with_retry(adapter, user_prompt, system, config, item, stats, sleep_fn)
            with lock:
                out_fh.write(json.dumps(_row_for(item_id, regime, sample_index, adapter.model_id, response), ensure_ascii=False) + "\n")
                out_fh.flush()
                stats.calls_made += 1
                if response.error:
                    stats.errors += 1

        if batch_size > 1:
            if not hasattr(adapter, "complete_batch"):
                raise TypeError(f"{type(adapter).__name__} has no complete_batch() - batch_size requires it")
            for i in range(0, len(pending), batch_size):
                chunk = pending[i : i + batch_size]
                requests = [(system, user_prompt) for *_rest, system, user_prompt in chunk]
                responses = adapter.complete_batch(requests, temperature=config.temperature, max_tokens=config.max_tokens)
                for (item_id, regime, sample_index, _item, _system, _prompt), response in zip(chunk, responses):
                    out_fh.write(json.dumps(_row_for(item_id, regime, sample_index, adapter.model_id, response), ensure_ascii=False) + "\n")
                    stats.calls_made += 1
                    if response.error:
                        stats.errors += 1
                out_fh.flush()
        elif max_workers <= 1:
            for work in pending:
                handle_one(work)
        else:
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                list(pool.map(handle_one, pending))
    return stats
