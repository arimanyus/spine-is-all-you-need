"""Teacher-forced scorers against a fake byte-level LM, so no weights or GPU are needed."""

from types import SimpleNamespace

import pytest

from scorers.spine import delta_logprob, delta_logprob_vs_clean, target_logprob

torch = pytest.importorskip("torch")

VOCAB, BIAS = 256, 5.0


def _ids(text):
    return torch.tensor([[ord(c) % VOCAB for c in text]])


class FakeTokenizer:
    def apply_chat_template(self, history, add_generation_prompt, return_tensors):
        return _ids("|".join(m["content"] for m in history) + "|")

    def __call__(self, text, add_special_tokens, return_tensors):
        return SimpleNamespace(input_ids=_ids(text))


class FakeLM:
    """Next-token logits: +BIAS on 'a', 0 elsewhere, regardless of context."""

    device = "cpu"

    def __call__(self, input_ids):
        logits = torch.zeros(1, input_ids.shape[1], VOCAB)
        logits[..., ord("a")] = BIAS
        return SimpleNamespace(logits=logits)


MODEL = SimpleNamespace(tokenizer=FakeTokenizer(), model=FakeLM())
HISTORY = [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}, {"role": "user", "content": "no, b"}]


def test_target_logprob_is_sum_over_target_tokens():
    one = target_logprob(MODEL, HISTORY, "a")
    assert target_logprob(MODEL, HISTORY, "aa") == pytest.approx(2 * one)


def test_contrastive_delta_prefers_gold_and_is_antisymmetric():
    d = delta_logprob(MODEL, HISTORY, "a", "b")
    assert d == pytest.approx(BIAS)
    assert delta_logprob(MODEL, HISTORY, "b", "a") == pytest.approx(-d)


def test_contrastive_delta_ignores_shared_format_term():
    # Identical targets => the whole format term cancels.
    assert delta_logprob(MODEL, HISTORY, "a", "a") == 0.0


def test_legacy_delta_is_zero_without_pressure_and_context_free_here():
    assert delta_logprob_vs_clean(MODEL, HISTORY[:1], "a") == 0.0
    assert delta_logprob_vs_clean(MODEL, HISTORY, "a") == pytest.approx(0.0)
