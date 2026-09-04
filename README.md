# BKP-500 — Bharat Knowledge Probe (eval harness)

*Does your model know where it is?*

[![License](https://img.shields.io/badge/code-Apache--2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](pyproject.toml)
[![Dataset](https://img.shields.io/badge/🤗%20dataset-sthanika--ai%2FBharat--Knowledge--Probe--Benchmark-yellow.svg)](https://huggingface.co/datasets/sthanika-ai/Bharat-Knowledge-Probe-Benchmark)
[![Results](https://img.shields.io/badge/results-sthanika--ai%2FBKP--500--model--runs-blue.svg)](https://github.com/sthanika-ai/BKP-500-model-runs)

This repository is the **evaluation harness** for [BKP-500](https://huggingface.co/datasets/sthanika-ai/Bharat-Knowledge-Probe-Benchmark),
a benchmark of things every Indian knows and frontier LLMs routinely fumble — lakh/crore
arithmetic, Indian digit grouping, state-specific land units (bigha, katha, guntha...), traditional
mass units, the Indian fiscal year, agricultural crop seasons, and structural identifiers (PAN,
GSTIN, IFSC, PIN codes).

The dataset itself — items, matched control twins, and per-item provenance — is hosted separately
on the Hugging Face Hub: **[sthanika-ai/Bharat-Knowledge-Probe-Benchmark](https://huggingface.co/datasets/sthanika-ai/Bharat-Knowledge-Probe-Benchmark)**.
This repo is the code that runs a model against it and grades the responses.

## The core idea: matched control twins

Every quantitative item has a **control twin** — arithmetically identical, but internationally
framed (USD/million instead of ₹/lakh-crore):

| | Item | Control twin |
|---|---|---|
| Prompt | "A scheme outlay is ₹1.2 lakh crore. Express it in USD billions at ₹83/USD." | "A regional budget is €13,200 million. Express it in USD billions at $1.10/€." |
| Measures | locale competence + arithmetic | arithmetic only |

**Locale Gap Δ = accuracy(control) − accuracy(India-framed).** That single number separates "bad
at math" from "doesn't know where it is."

## What's in this repo

| Path | What it is |
|---|---|
| `src/bkp_eval/` | The harness: model adapters (OpenAI, Anthropic, HF `transformers`, local OpenAI-compatible servers), graders (numeric, categorical, date), and scoring. |
| `src/bharat_units/` | A small normalizer library (Indian numerals, mass, fiscal year, seasons) that the numeric/categorical graders use as the reference implementation of "correct" — the same library the dataset's gold answers were computed against. |
| `scripts/run_eval.py` | CLI entrypoint: wires a model adapter + the corpus + the runner together. |
| `tests/` | Unit and property-based tests for the harness and `bharat_units`. |

No benchmark data ships in this repo — see [Getting the dataset](#getting-the-dataset) below.

## Install

```bash
git clone https://github.com/sthanika-ai/Bharat-Knowledge-Probe-Benchmark.git
cd Bharat-Knowledge-Probe-Benchmark
pip install -e ".[dev]"
```

Requires Python ≥3.10. One extra is optional, only needed if you want to load the corpus
straight from the Hub instead of a local checkout (see below):

```bash
pip install -e ".[hf-datasets]"
```

## Getting the dataset

Either clone the dataset repo into `data/` (matches the on-disk layout `load_corpus()` expects
by default):

```bash
git clone https://huggingface.co/datasets/sthanika-ai/Bharat-Knowledge-Probe-Benchmark data
```

...or load it straight from the Hub in Python, with no local checkout:

```python
from bkp_eval.items import load_corpus_from_hf

corpus = load_corpus_from_hf()   # sthanika-ai/Bharat-Knowledge-Probe-Benchmark, every category config + controls
```

## Running an evaluation

```bash
python scripts/run_eval.py --model <name> --mode smoke   # small sanity run
python scripts/run_eval.py --model <name> --mode full
```

Local/self-hosted models (vLLM, Ollama) need no API key — the harness talks to them over a local
OpenAI-compatible endpoint via `LocalAPIAdapter`. If you evaluate a model through the OpenAI or
Anthropic cloud APIs instead, `pip install openai` / `pip install anthropic` and export
`OPENAI_API_KEY` / `ANTHROPIC_API_KEY` in your shell before running (see
`src/bkp_eval/adapters/openai_adapter.py` / `anthropic_adapter.py` for exactly what they read).
See `scripts/run_eval.py` for the model registry
and per-model config (device, batch size, sampling), and `src/bkp_eval/adapters/` to wire up a new
backend.

### Scoring a results file directly

```bash
python -m bkp_eval.score results/<model>/full.jsonl
```

reports Bharat Score, per-category accuracy, Locale Gap Δ, and secondary metrics (unit discipline,
overconfidence, refusal rate, consistency across repeated samples).

## Status & caveats

This is an active work in progress, not a finished v1 release.

- **No public/gated split yet.** All items ship in the `draft` split with gold answers visible —
  treat current scores as provisional until a contamination-resistant public-dev / gated-test
  split lands. See the dataset card for details.
- **`bharat-units`'s CLI is a stub** — use the library modules directly (`bharat_units.numerals`,
  `.mass`, `.fiscal`, `.seasons`, `.identifiers`).
- **Cloud adapters are implemented but not yet run against a live key** (`openai_adapter.py`,
  `anthropic_adapter.py`) — see their docstrings before relying on them for a paid run.

## Testing

```bash
pytest
ruff check .
mypy src
```

## Citation

```bibtex
@misc{bkp500,
  title  = {Bharat Knowledge Probe (BKP-500): Does Your Model Know Where It Is?},
  author = {{Sthanika AI}},
  year   = {2026},
  url    = {https://github.com/sthanika-ai/Bharat-Knowledge-Probe-Benchmark}
}
```

See [`CITATION.cff`](CITATION.cff) for a machine-readable citation record.

## License

- **Harness code** in this repository: [Apache-2.0](LICENSE).
- **Dataset** (items, controls, provenance): CC-BY-NC-4.0 — see the
  [dataset card](https://huggingface.co/datasets/sthanika-ai/Bharat-Knowledge-Probe-Benchmark) for the exact terms.

## Contributing

Issues and PRs are welcome — bug reports on graders/adapters, new model adapters, and corrections
to `bharat_units`'s reference facts (with a source) are all in scope. Corrections or additions to
the benchmark *data itself* (new items, fixed gold answers, sourcing) belong on the
[dataset repo](https://huggingface.co/datasets/sthanika-ai/Bharat-Knowledge-Probe-Benchmark), not here.

## Related repositories

- **[BKP-500 dataset](https://huggingface.co/datasets/sthanika-ai/Bharat-Knowledge-Probe-Benchmark)** — the corpus itself, on Hugging Face
- **[BKP-500-model-runs](https://github.com/sthanika-ai/BKP-500-model-runs)** — raw responses, scored reports, and exact run configuration for every model evaluated with this harness
