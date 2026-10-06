"""Loading the BKP-500 item corpus into memory.

Every function here is read-only and side-effect free - grading and scoring both start from
`load_corpus()` (local JSONL files) or `load_corpus_from_hf()` (the published dataset,
https://huggingface.co/datasets/sthanika-ai/Bharat-Knowledge-Probe-Benchmark), so there is exactly one place that knows
each on-disk/on-hub layout.

This package ships no data of its own - point `load_corpus()` at a local checkout of the dataset
(e.g. `git clone https://huggingface.co/datasets/sthanika-ai/Bharat-Knowledge-Probe-Benchmark data`) or call
`load_corpus_from_hf()` to pull it straight from the Hub via the `datasets` library.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_ITEMS_DIR = REPO_ROOT / "data" / "items"
DEFAULT_HF_REPO_ID = "sthanika-ai/Bharat-Knowledge-Probe-Benchmark"


def _read_jsonl(path: Path) -> list[dict]:
    records = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


@dataclass(frozen=True)
class Corpus:
    """Every BKP-500 item, indexed by id."""

    items: dict[str, dict]

    @property
    def all_by_id(self) -> dict[str, dict]:
        """Alias for `items`. Kept so callers written back when a separate data/controls/
        controls.jsonl existed (and "look this id up regardless of which file it came from" was
        a real distinction) don't need to change - now it's just `items` under another name.
        """
        return self.items


def load_corpus(items_dir: Path | str = DEFAULT_ITEMS_DIR) -> Corpus:
    """Load every data/items/*.jsonl file."""
    items_dir = Path(items_dir)
    items: dict[str, dict] = {}
    for path in sorted(items_dir.glob("*.jsonl")):
        for record in _read_jsonl(path):
            items[record["id"]] = record

    return Corpus(items=items)


def load_corpus_from_hf(repo_id: str = DEFAULT_HF_REPO_ID, revision: str | None = None) -> Corpus:
    """Load the corpus straight from the Hugging Face Hub - no local checkout required.

    Requires the optional `datasets` dependency (`pip install "bkp-eval[hf-datasets]"` or plain
    `pip install datasets`). Iterates every config the dataset repo publishes (one per item
    category) rather than hardcoding category names here, so a new category added on the Hub
    side needs no change on this side.
    """
    from datasets import get_dataset_config_names, load_dataset

    items: dict[str, dict] = {}
    for config in get_dataset_config_names(repo_id, revision=revision):
        if config in ("all_items", "default"):
            continue  # convenience aggregate configs - every category config already covers this
        ds = load_dataset(repo_id, config, split="train", revision=revision)
        for record in ds:
            items[record["id"]] = dict(record)

    return Corpus(items=items)
