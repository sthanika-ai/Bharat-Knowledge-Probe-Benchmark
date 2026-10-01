# BKP-500 — Bharat Knowledge Probe

Evaluation harness for BKP-500, a benchmark of whether language models know India-specific locale conventions.

[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Dataset](https://img.shields.io/badge/%F0%9F%A4%97%20dataset-BKP--500-yellow)](https://huggingface.co/datasets/sthanika-ai/Bharat-Knowledge-Probe-Benchmark)
[![Report](https://img.shields.io/badge/report-sthanika.ai-black)](https://sthanika.ai/research/bkp500-2026)

## What it measures

BKP-500 tests knowledge of Indian numeral magnitudes and digit grouping (lakh/crore), state-specific land units (bigha, katha, guntha), traditional mass units, the Indian fiscal year, agricultural crop seasons, government schemes and structural identifiers (PAN, GSTIN, IFSC, PIN codes). Scoring is fully deterministic, with no LLM judge. This repo is the code that runs a model against the corpus and grades the responses: model adapters (OpenAI, Anthropic, HF `transformers`, local OpenAI-compatible servers), numeric, categorical and date graders, and scoring, plus `bharat_units`, the reference normalizer library the gold answers were computed against. No benchmark data ships here. Benchmark page: [sthanika.ai](https://sthanika.ai) <!-- TODO: link the exact BKP-500 page -->

## Quickstart

Requires Python 3.10 or newer. The corpus is auto-gated on Hugging Face: accept its terms on the dataset page, then log in once.

```bash
git clone https://github.com/sthanika-ai/Bharat-Knowledge-Probe-Benchmark.git
cd Bharat-Knowledge-Probe-Benchmark
pip install -e ".[dev]" huggingface_hub
huggingface-cli login

# Corpus -> data/items/*.jsonl, the layout load_corpus() reads by default
python -c "from huggingface_hub import snapshot_download; snapshot_download('sthanika-ai/Bharat-Knowledge-Probe-Benchmark', repo_type='dataset', local_dir='.', allow_patterns=['data/items/*.jsonl'])"

# Confirm the install is sound
pytest

# Smoke test, then a full run (3 samples, both regimes), then score it
python scripts/run_eval.py --model <name> --mode smoke
python scripts/run_eval.py --model <name> --mode full
python -m bkp_eval.score results/<name>/full.jsonl
```

Notes:

- `<name>` is a key in the `MODELS` registry. The registry in this repo's `scripts/run_eval.py` is a template; the exact run script and per-model configs for the 19 published models are in [BKP-500-model-runs](https://github.com/sthanika-ai/BKP-500-model-runs).
- Self-hosted models (vLLM, Ollama) need no API key. The harness talks to them over a local OpenAI-compatible endpoint via `LocalAPIAdapter`. For OpenAI or Anthropic models, `pip install openai` or `pip install anthropic` and export `OPENAI_API_KEY` or `ANTHROPIC_API_KEY`. The cloud adapters are implemented but have not been run against a live key.
- To load the corpus from the Hub in Python instead: `pip install -e ".[hf-datasets]"`, then `from bkp_eval.items import load_corpus_from_hf`.
- `python -m bkp_eval.score` reports Bharat Score, per-category accuracy, unit discipline, overconfidence, refusal rate and consistency across repeated samples.
- `bharat_units`' CLI is a stub; use the library modules directly (`bharat_units.numerals`, `.mass`, `.fiscal`, `.seasons`, `.identifiers`).
- Lint and type checks: `ruff check .` and `mypy src`.

## Results

Full report: [sthanika.ai/research/bkp500-2026](https://sthanika.ai/research/bkp500-2026). Exact run configuration and leaderboard for 19 models: [BKP-500-model-runs](https://github.com/sthanika-ai/BKP-500-model-runs).

## Citation

Cite as in [`CITATION.cff`](CITATION.cff).

```bibtex
@misc{bkp500,
  title  = {Bharat Knowledge Probe (BKP-500): Does Your Model Know Where It Is?},
  author = {{Sthanika AI}},
  year   = {2026},
  url    = {https://github.com/sthanika-ai/Bharat-Knowledge-Probe-Benchmark}
}
```

## License

Apache-2.0 for the harness code, see [LICENSE](LICENSE). The dataset is CC-BY-NC-4.0; see the [dataset card](https://huggingface.co/datasets/sthanika-ai/Bharat-Knowledge-Probe-Benchmark).

## Related

- [BKP-500 dataset](https://huggingface.co/datasets/sthanika-ai/Bharat-Knowledge-Probe-Benchmark): the corpus, on Hugging Face. Corrections to the data itself belong there.
- [BKP-500-model-runs](https://github.com/sthanika-ai/BKP-500-model-runs): run configuration and leaderboard for every evaluated model
- Site: [sthanika.ai](https://sthanika.ai)
