"""The grading cascade dispatcher: deterministic normalizer-based grader -> human.

`grade()` is the one entry point everything else (score.py, tests) should call - it picks the
right deterministic grader off `item["grader"]` and returns its verdict. Anything the
deterministic grader declines to rule on comes back with `correct=None` for a human to resolve;
score.py buckets those separately (undetermined_rate) rather than counting them as wrong.

This originally had a third cascade tier here - an LLM-judge, scoped to clarification/
string_normalized items only - between the deterministic grader and human review.
It was removed (schema.json x-changelog 0.1.4): it was never wired to a real judge model across
any run (0 judge calls ever made, by construction it always declined), and no item in the corpus
has ever used grader='judge_v1'. Re-grading every existing run's results confirmed removing it
changes zero verdicts - see bkp_eval/graders/judge.py in version control history if a real
LLM-judge tier is ever built again.
"""
from __future__ import annotations

from bkp_eval.graders._common import GradeResult
from bkp_eval.graders.categorical import grade_clarification, grade_enum, grade_month_set, grade_string_normalized
from bkp_eval.graders.dates import grade_date, grade_date_range
from bkp_eval.graders.numeric import grade_numeric, grade_numeric_with_unit

__all__ = ["GradeResult", "grade", "DETERMINISTIC_GRADERS"]

# grader id (schema.json's `grader` enum) -> deterministic grading function.
DETERMINISTIC_GRADERS = {
    "numeric_v1": grade_numeric,
    "numeric_unit_v1": grade_numeric_with_unit,
    "date_v1": grade_date,
    "date_range_v1": grade_date_range,
    "enum_v1": grade_enum,
    "string_normalized_v1": grade_string_normalized,
    "month_set_v1": grade_month_set,
    "clarification_v1": grade_clarification,
}


def grade(item: dict, raw_response: str, regime: str = "R2") -> GradeResult:
    grader_id = item["grader"]
    deterministic = DETERMINISTIC_GRADERS.get(grader_id)
    if deterministic is None:
        raise ValueError(f"no grader registered for grader id {grader_id!r} (item {item['id']!r})")
    return deterministic(item, raw_response, regime)
