"""bkp_eval — evaluation harness for the Bharat Knowledge Probe (BKP-500) benchmark.

Wires together: model adapters (adapters/ - OpenAI, Anthropic, HF transformers, local
OpenAI-compatible servers), a resumable rate-limit-aware runner (runner.py) that calls those
adapters against the item corpus, a deterministic grading cascade (graders/) that scores each
response, and a metrics aggregator (score.py) that turns graded results into Bharat Score,
per-category accuracy, and Locale Gap Delta.
"""

__version__ = "0.1.0"
