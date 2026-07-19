from router.grade import grade_gsm8k, grade_mmlu


def test_gsm8k_exact():
    assert grade_gsm8k("...so the total is 42.\nAnswer: 42", "42") is True


def test_gsm8k_with_commas_and_dollars():
    assert grade_gsm8k("Answer: $1,234", "1234") is True


def test_gsm8k_wrong():
    assert grade_gsm8k("Answer: 41", "42") is False


def test_gsm8k_no_answer_line_falls_back_to_last_number():
    assert grade_gsm8k("The result is 42.", "42") is True


def test_gsm8k_decimal_equivalence():
    assert grade_gsm8k("Answer: 42.0", "42") is True


def test_gsm8k_garbage():
    assert grade_gsm8k("I cannot solve this.", "42") is False


def test_mmlu_letter():
    assert grade_mmlu("Answer: B", "B") is True


def test_mmlu_lowercase_and_parens():
    assert grade_mmlu("the answer is (c)", "C") is True


def test_mmlu_wrong():
    assert grade_mmlu("Answer: A", "D") is False


def test_mmlu_no_letter():
    assert grade_mmlu("Both options seem plausible.", "A") is False
