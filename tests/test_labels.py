from router.labels import label_battle

TIERS = {"gpt-4-1106-preview": 0, "claude-2.1": 1, "mixtral-8x7b-instruct-v0.1": 2, "llama-2-7b-chat": 5}
STRONG = {0, 1}
WEAK = {2}


def test_strong_side_a_wins():
    lab = label_battle("gpt-4-1106-preview", "mixtral-8x7b-instruct-v0.1", "model_a", TIERS, STRONG, WEAK)
    assert lab == 1


def test_weak_side_wins():
    lab = label_battle("mixtral-8x7b-instruct-v0.1", "claude-2.1", "model_a", TIERS, STRONG, WEAK)
    assert lab == 0


def test_strong_side_b_wins():
    lab = label_battle("mixtral-8x7b-instruct-v0.1", "gpt-4-1106-preview", "model_b", TIERS, STRONG, WEAK)
    assert lab == 1


def test_same_class_battle_dropped():
    assert label_battle("gpt-4-1106-preview", "claude-2.1", "model_a", TIERS, STRONG, WEAK) is None


def test_out_of_class_model_dropped():
    assert label_battle("gpt-4-1106-preview", "llama-2-7b-chat", "model_a", TIERS, STRONG, WEAK) is None


def test_tie_dropped():
    assert label_battle("gpt-4-1106-preview", "mixtral-8x7b-instruct-v0.1", "tie", TIERS, STRONG, WEAK) is None


def test_unknown_model_dropped():
    assert label_battle("gpt-4-1106-preview", "some-new-model", "model_a", TIERS, STRONG, WEAK) is None
