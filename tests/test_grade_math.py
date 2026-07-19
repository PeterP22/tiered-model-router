from router.grade import grade_math, grade_mmlu


def test_math_plain_number():
    assert grade_math("...\nAnswer: 42", "42") is True


def test_math_fraction_exact():
    assert grade_math("Answer: \\frac{1}{2}", "\\frac{1}{2}") is True


def test_math_dfrac_normalizes_to_frac():
    assert grade_math("Answer: \\dfrac{1}{2}", "\\frac{1}{2}") is True


def test_math_boxed_fallback():
    assert grade_math("The result is $\\boxed{3\\sqrt{2}}$.", "3\\sqrt{2}") is True


def test_math_dollar_signs_stripped():
    assert grade_math("Answer: $12$", "12") is True


def test_math_wrong():
    assert grade_math("Answer: 41", "42") is False


def test_math_numeric_equivalence():
    assert grade_math("Answer: 42.0", "42") is True


def test_math_no_answer():
    assert grade_math("I ran out of steps.", "42") is False


def test_math_last_answer_line_wins():
    out = "Answer: 5 is wrong, let me redo.\n...\nAnswer: 7"
    assert grade_math(out, "7") is True


def test_mmlu_pro_letters_beyond_d():
    assert grade_mmlu("Answer: J", "J") is True
    assert grade_mmlu("Answer: G", "J") is False
