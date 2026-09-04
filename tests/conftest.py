"""Session-wide test collection rules.

Several test modules (graders, adapters/mock, runner, score) validate the harness against the
REAL BKP-500 corpus rather than synthetic fixtures - e.g.
`test_graders.py::test_gold_answer_always_grades_correct` runs every real item+control through
`grade()` against its own gold answer, on the theory that a grader bug that rejects a right
answer is exactly the kind of thing a hand-picked fixture would miss.

That corpus is published separately (https://huggingface.co/datasets/sthanika-ai/Bharat-Knowledge-Probe-Benchmark) and
isn't vendored into this code repo - see the README's "Getting the dataset" section. On a fresh
checkout that hasn't fetched it into `data/` yet, skip those modules with a clear message rather
than fail on a confusing FileNotFoundError deep in `bkp_eval.items`.
"""
from __future__ import annotations

from bkp_eval.items import DEFAULT_CONTROLS_PATH, DEFAULT_ITEMS_DIR

_CORPUS_DEPENDENT = [
    "graders/test_graders.py",
    "adapters/test_mock.py",
    "runner/test_runner.py",
    "test_score.py",
]

collect_ignore_glob: list[str] = []

if not DEFAULT_CONTROLS_PATH.exists() or not any(DEFAULT_ITEMS_DIR.glob("*.jsonl")):
    collect_ignore_glob.extend(_CORPUS_DEPENDENT)


def pytest_configure(config) -> None:
    if collect_ignore_glob:
        print(
            "\n[bkp-eval] Skipping corpus-dependent tests (no data/ checkout found) - see "
            "README's 'Getting the dataset' to run the full suite: "
            "git clone https://huggingface.co/datasets/sthanika-ai/Bharat-Knowledge-Probe-Benchmark data\n"
        )
