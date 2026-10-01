# BKP-500 — Bharat Knowledge Probe

Evaluation harness for BKP-500, a benchmark of whether language models know India-specific locale conventions.

[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-56BF4F?style=flat-square&labelColor=1E281F)](LICENSE)
[![Hugging Face dataset](https://img.shields.io/badge/%F0%9F%A4%97%20dataset-BKP--500-FFD21E?style=flat-square&labelColor=1E281F)](https://huggingface.co/datasets/sthanika-ai/Bharat-Knowledge-Probe-Benchmark)
[![Report](https://img.shields.io/badge/report-sthanika.ai-56BF4F?style=flat-square&labelColor=1E281F&logo=firefox&logoColor=white)](https://sthanika.ai/research/bkp500-2026)

## What it measures

BKP-500 tests knowledge of Indian numeral magnitudes and digit grouping (lakh/crore), state-specific land units (bigha, katha, guntha), traditional mass units, the Indian fiscal year, agricultural crop seasons, government schemes and structural identifiers (PAN, GSTIN, IFSC, PIN codes). Scoring is fully deterministic, with no LLM judge. This repo is the code that runs a model against the corpus and grades the responses: model adapters (OpenAI, Anthropic, HF `transformers`, local OpenAI-compatible servers), numeric, categorical and date graders, and scoring, plus `bharat_units`, the reference normalizer library the gold answers were computed against. No benchmark data ships here. Benchmark page: [sthanika.ai/benchmarks/bkp-500](https://sthanika.ai/benchmarks/bkp-500)

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

- `--model` must be a key in `MODELS` inside `run_eval.py`. The registry in this repo's `scripts/run_eval.py` is a template; the exact run script and per-model configs for the 19 published models are in [BKP-500-model-runs](https://github.com/sthanika-ai/BKP-500-model-runs).
- Self-hosted models (vLLM, Ollama) need no API key. The harness talks to them over a local OpenAI-compatible endpoint via `LocalAPIAdapter`. For OpenAI or Anthropic models, `pip install openai` or `pip install anthropic` and export `OPENAI_API_KEY` or `ANTHROPIC_API_KEY`. The cloud adapters are implemented but have not been run against a live key.
- To load the corpus from the Hub in Python instead: `pip install -e ".[hf-datasets]"`, then `from bkp_eval.items import load_corpus_from_hf`.
- `python -m bkp_eval.score` reports Bharat Score, per-category accuracy, unit discipline, overconfidence, refusal rate and consistency across repeated samples.
- `bharat_units`' CLI is a stub; use the library modules directly (`bharat_units.numerals`, `.mass`, `.fiscal`, `.seasons`, `.identifiers`).
- Lint and type checks: `ruff check .` and `mypy src`.

## Results

Full report: [sthanika.ai/research/bkp500-2026](https://sthanika.ai/research/bkp500-2026) · Model runs: [BKP-500-model-runs](https://github.com/sthanika-ai/BKP-500-model-runs). Ranked by Bharat Score (macro-mean accuracy across all 7 categories), tiered by chained bootstrap-CI overlap.

| tier | model | bucket | bharat score (95% CI) | oom error rate | unit discipline | refusal rate | consistency |
|---|---|---|---|---|---|---|---|
| 1 | Qwen3.6 27B | global | 53.2% (48.9–57.3) | 10.6% | 84.9% | 0.7% | 100.0% |
| 1 | gpt-oss-20b | global | 46.9% (43.0–50.9) | 16.5% | 84.6% | 0.1% | 87.6% |
| 1 | Sarvam-M 24B | india | 45.2% (40.9–49.2) | 22.2% | 89.1% | 3.1% | 90.7% |
| 1 | Gemma 3 27B | global | 41.8% (38.3–45.8) | 18.4% | 85.3% | 0.0% | 99.6% |
| 1 | Qwen3-VL 8B | global | 39.2% (35.1–43.4) | 20.5% | 80.8% | 0.0% | 97.1% |
| 1 | Mistral Small 3.1 24B | global | 38.8% (35.0–42.9) | 17.1% | 68.8% | 0.6% | 97.7% |
| 1 | Krutrim-2 12B | india | 38.7% (34.6–43.3) | 24.1% | 82.3% | 0.2% | 99.5% |
| 1 | Phi-4 14B | global | 38.7% (34.8–43.0) | 21.2% | 81.6% | 2.1% | 99.0% |
| 1 | Sarvam 30B | india | 37.1% (33.5–40.9) | 33.7% | 90.5% | 1.3% | 86.7% |
| 1 | Gemma 3 12B | global | 35.1% (31.3–39.4) | 25.6% | 83.2% | 0.0% | 99.6% |
| 1 | Llama 3.1 8B Instruct | global | 32.6% (28.5–36.7) | 32.9% | 77.9% | 2.9% | 100.0% |
| 1 | Qwen2.5 7B | global | 30.9% (27.1–34.8) | 23.1% | 82.9% | 0.1% | 97.9% |
| 1 | Gemma 3 4B | global | 25.3% (22.1–28.8) | 33.7% | 75.9% | 0.0% | 100.0% |
| 1 | Navarasa 2.0 7B | india | 20.6% (17.8–23.6) | 44.3% | 74.7% | 0.6% | 93.1% |
| 2 | Param-1 2.9B | india | 14.3% (11.8–16.9) | 54.2% | 60.4% | 0.6% | 99.7% |
| 2 | Sarvam-1 2B | india | 11.6% (9.1–14.8) | 56.7% | 47.5% | 1.5% | 99.7% |
| 2 | OpenHathi 7B | india | 11.2% (8.7–14.2) | 52.3% | 65.8% | 0.1% | 96.1% |
| 2 | Airavata 7B | india | 7.9% (5.6–10.4) | 22.7% | 14.5% | 38.5% | 99.9% |
| 2 | Gemma 3 12B INT4 | global | 4.6% (2.4–7.2) | 1.4% | 2.0% | 91.3% | 100.0% |

OOM error rate is the share of numeric answers off by 10× or more (a lakh/crore-scale slip). Unit discipline is the share of `numeric_with_unit` answers that included an explicit, correct unit. CIs use n_resamples=2,000 (the harness default for a publication-grade run is 10,000).

Per-category accuracy (%), all 19 models × 7 categories:

| model | numerals | weights/vol | land units | seasons | fiscal yr | schemes | identifiers |
|---|---|---|---|---|---|---|---|
| Qwen3.6 27B | 77.9 | 61.1 | 39.4 | 48.9 | 52.9 | 41.9 | 50.0 |
| gpt-oss-20b | 78.7 | 57.6 | 34.9 | 41.2 | 46.9 | 25.4 | 43.3 |
| Sarvam-M 24B | 46.6 | 55.8 | 21.5 | 54.6 | 49.0 | 36.5 | 52.2 |
| Gemma 3 27B | 54.0 | 54.9 | 26.4 | 38.5 | 50.6 | 36.0 | 32.2 |
| Qwen3-VL 8B | 55.4 | 39.4 | 31.2 | 42.1 | 39.8 | 29.7 | 36.7 |
| Mistral Small 3.1 24B | 50.6 | 59.0 | 26.1 | 42.9 | 32.2 | 31.1 | 30.0 |
| Krutrim-2 12B | 49.2 | 39.6 | 24.5 | 44.7 | 42.9 | 39.9 | 30.0 |
| Phi-4 14B | 46.1 | 53.5 | 24.2 | 42.1 | 46.1 | 32.2 | 26.7 |
| Sarvam 30B | 40.7 | 51.2 | 28.8 | 51.1 | 33.1 | 27.9 | 26.7 |
| Gemma 3 12B | 48.8 | 40.7 | 20.8 | 41.9 | 37.8 | 29.0 | 26.7 |
| Llama 3.1 8B Instruct | 32.0 | 25.7 | 17.3 | 39.6 | 38.2 | 35.1 | 40.0 |
| Qwen2.5 7B | 44.6 | 37.7 | 17.5 | 32.4 | 34.5 | 16.0 | 33.3 |
| Gemma 3 4B | 32.9 | 24.3 | 12.5 | 34.6 | 31.2 | 18.2 | 23.3 |
| Navarasa 2.0 7B | 26.1 | 21.5 | 10.3 | 25.1 | 30.4 | 17.6 | 13.3 |
| Param-1 2.9B | 15.2 | 11.1 | 3.5 | 25.8 | 15.3 | 15.5 | 13.3 |
| Sarvam-1 2B | 16.2 | 2.8 | 2.4 | 18.5 | 12.8 | 14.9 | 13.3 |
| OpenHathi 7B | 13.1 | 4.9 | 2.2 | 15.4 | 16.7 | 8.8 | 17.8 |
| Airavata 7B | 6.9 | 4.9 | 1.4 | 12.6 | 11.8 | 4.0 | 13.3 |
| Gemma 3 12B INT4 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 12.2 | 20.0 |

Qwen3.6 27B, a global open-weight model, leads the roster. Sarvam-M 24B is the strongest India-built model at rank 3. Land units by state is the weakest category for 14 of the 19 models.

Exact run configuration for every model: [BKP-500-model-runs](https://github.com/sthanika-ai/BKP-500-model-runs) · Report page: [sthanika.ai](https://sthanika.ai/research/bkp500-2026)

## Citation

Cite the benchmark and this harness as in [`CITATION.cff`](CITATION.cff). To cite the leaderboard and run configurations, cite the [BKP-500-model-runs](https://github.com/sthanika-ai/BKP-500-model-runs) repository.

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
