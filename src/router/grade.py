"""Deterministic graders for golden-label tasks.

Pure functions: (model_output, gold_answer) -> bool. Kept intentionally strict
and simple; grader failures count as wrong answers, which is the conservative
direction for a router label ("cheap model didn't demonstrably solve it").
"""

from __future__ import annotations

import re

_NUM = re.compile(r"-?\$?\d[\d,]*\.?\d*")
_LETTER = re.compile(r"\b\(?([A-Da-d])\)?\b")


def _norm_number(s: str) -> float | None:
    try:
        return float(s.replace("$", "").replace(",", ""))
    except ValueError:
        return None


def grade_gsm8k(output: str, gold: str) -> bool:
    """Match the number on the final 'Answer:' line, else the last number."""
    target = _norm_number(gold)
    if target is None:
        return False
    m = re.search(r"[Aa]nswer:\s*(" + _NUM.pattern + ")", output)
    if m:
        return _norm_number(m.group(1)) == target
    nums = _NUM.findall(output)
    return bool(nums) and _norm_number(nums[-1]) == target


def grade_mmlu(output: str, gold: str) -> bool:
    """Match the letter on an 'Answer:' line, else the last standalone A-D."""
    m = re.search(r"[Aa]nswer[^A-Da-d]{0,5}([A-Da-d])", output)
    if m:
        return m.group(1).upper() == gold.upper()
    letters = _LETTER.findall(output)
    return bool(letters) and letters[-1].upper() == gold.upper()
