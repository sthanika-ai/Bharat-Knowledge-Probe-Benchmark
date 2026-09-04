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
DEFAULT_CONTROLS_PATH = REPO_ROOT / "data" / "controls" / "controls.jsonl"
DEFAULT_HF_REPO_ID = "sthanika-ai/Bharat-Knowledge-Probe-Benchmark"

# answer_types that are required (per schema.json) to carry a matched control twin.
PAIRED_ANSWER_TYPES = frozenset({"numeric", "numeric_with_unit", "date", "date_range"})


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
    """All items + all controls, indexed by id. `items` excludes controls; `controls` excludes
    core items - use `all_by_id` when you just need to look something up by id regardless of
    which file it came from (that's what graders/score.py normally want).
    """

    items: dict[str, dict]
    controls: dict[str, dict]

    @property
    def all_by_id(self) -> dict[str, dict]:
        return {**self.items, **self.controls}

    def pairs(self) -> list[tuple[dict, dict]]:
        """(core_item, control_item) for every core item that has a matched twin.

        Every PAIRED_ANSWER_TYPES item has one by construction (validate_items.py / schema.json
        enforce it); a handful of other-typed items were also given twins (see control_id in
        schema.json) and are included here too, since they're just as usable for Delta.
        """
        out = []
        for item in self.items.values():
            control_id = item.get("control_id")
            if control_id is None:
                continue
            control = self.controls.get(control_id)
            if control is None:
                raise KeyError(
                    f"{item['id']} points at control_id {control_id!r}, which isn't in controls.jsonl"
                )
            out.append((item, control))
        return out


def load_corpus(
    items_dir: Path | str = DEFAULT_ITEMS_DIR,
    controls_path: Path | str = DEFAULT_CONTROLS_PATH,
) -> Corpus:
    """Load every data/items/*.jsonl file plus data/controls/controls.jsonl."""
    items_dir = Path(items_dir)
    items: dict[str, dict] = {}
    for path in sorted(items_dir.glob("*.jsonl")):
        for record in _read_jsonl(path):
            items[record["id"]] = record

    controls: dict[str, dict] = {}
    for record in _read_jsonl(Path(controls_path)):
        controls[record["id"]] = record

    return Corpus(items=items, controls=controls)


def load_corpus_from_hf(repo_id: str = DEFAULT_HF_REPO_ID, revision: str | None = None) -> Corpus:
    """Load the corpus straight from the Hugging Face Hub - no local checkout required.

    Requires the optional `datasets` dependency (`pip install "bkp-eval[hf-datasets]"` or plain
    `pip install datasets`). Iterates every config the dataset repo publishes (one per item
    category, plus a `controls` config) rather than hardcoding category names here, so a new
    category added on the Hub side needs no change on this side.
    """
    from datasets import get_dataset_config_names, load_dataset

    items: dict[str, dict] = {}
    controls: dict[str, dict] = {}
    for config in get_dataset_config_names(repo_id, revision=revision):
        if config in ("all_items", "default"):
            continue  # convenience aggregate configs - every category config already covers this
        ds = load_dataset(repo_id, config, split="train", revision=revision)
        bucket = controls if config == "controls" else items
        for record in ds:
            bucket[record["id"]] = dict(record)

    return Corpus(items=items, controls=controls)
