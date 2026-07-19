"""Deterministic graders for golden-label tasks.

Pure functions: (model_output, gold_answer) -> bool. Kept intentionally strict
and simple; grader failures count as wrong answers, which is the conservative
direction for a router label ("cheap model didn't demonstrably solve it").
"""

from __future__ import annotations

import re

_NUM = re.compile(r"-?\$?\d[\d,]*\.?\d*")
_LETTER = re.compile(r"\b\(?([A-J])\)?\b")
_BOXED = re.compile(r"\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}")


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
    """Match the letter on an 'Answer:' line, else the last standalone A-J.

    Covers both MMLU (A-D) and MMLU-Pro (A-J); the fallback only matches
    uppercase letters so prose words like "a" don't false-positive.
    """
    m = re.search(r"[Aa]nswer(?:\s+is)?\s*[:\-]?\s*\(?([A-Ja-j])\)?(?![A-Za-z])", output)
    if m:
        return m.group(1).upper() == gold.upper()
    letters = _LETTER.findall(output)
    return bool(letters) and letters[-1].upper() == gold.upper()


def _norm_math(s: str) -> str:
    s = s.strip().strip("$").strip()
    s = s.replace("\\left", "").replace("\\right", "")
    s = s.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    s = s.replace("\\!", "").replace("\\,", "").replace(" ", "")
    s = s.rstrip(".")
    try:
        f = float(s.replace(",", ""))
        return str(int(f)) if f == int(f) else str(f)
    except ValueError:
        return s


def grade_math(output: str, gold: str) -> bool:
    """MATH-style grading: last 'Answer:' line, else last \\boxed{...}.

    Normalized string equality — conservative (misses some algebraically
    equivalent forms), but both models are graded by the same rule so the
    capability GAP stays meaningful.
    """
    lines = re.findall(r"[Aa]nswer:\s*(.+)", output)
    cand = lines[-1].strip() if lines else None
    if cand is None:
        boxed = _BOXED.findall(output)
        cand = boxed[-1] if boxed else None
    if cand is None:
        return False
    inner = _BOXED.findall(cand)
    if inner:
        cand = inner[-1]
    return _norm_math(cand) == _norm_math(gold)
