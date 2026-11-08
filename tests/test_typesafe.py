"""The Jev client's request shape, answer checks, retries and budget (scripted, no API calls)."""

import pytest

from shared.models.typesafe import MODEL, JevClient, TransientError


def test_noul_request_and_answer(tmp_path):
    sent = []

    def transport(request):
        sent.append(request)
        return {"model": "jev-1.13.0", "answers": {"q": {"type": "noul", "noul": 0.8}},
                "usage": {"input_tokens": 40, "output_tokens": 1}}
    client = JevClient(transport=transport, log_path=tmp_path / "jev.jsonl")
    assert client.noul("Sally put the ball in the basket.", "Sally thinks the ball is in the basket.") == 0.8
    assert sent[0] == {"model": MODEL, "state": "Sally put the ball in the basket.",
                       "questions": {"q": {"type": "noul", "instructions": "Sally thinks the ball is in the basket."}}}
    assert client.usage() == {"jev_calls": 1, "jev_input_tokens": 40, "jev_models": ["jev-1.13.0"]}
    assert (tmp_path / "jev.jsonl").read_text().count("\n") == 1


def test_choice_request_uses_named_criteria():
    sent = []
    options = {"low": "A simple model", "high": "A rich model"}

    def transport(request):
        sent.append(request)
        return {"model": "m", "answers": {"q": {"type": "choice", "choice": "high", "confidence": 0.6,
                                                "probabilities": {"low": 0.4, "high": 0.6}}}}
    answer = JevClient(transport=transport).choice("story", "Which model?", options)
    assert sent[0]["questions"]["q"] == {"type": "choice", "instructions": "Which model?", "criteria": options}
    assert answer["probabilities"]["high"] == 0.6


@pytest.mark.parametrize("answer", [{"type": "choice", "choice": "low", "probabilities": {"low": 1.0}},
                                    {"type": "choice", "choice": "mid", "probabilities": {"low": 0.5, "high": 0.5}},
                                    {"type": "score", "probabilities": {"0": 0.5, "1": 0.5}}])
def test_invalid_choice_answers_are_rejected(answer):
    client = JevClient(transport=lambda request: {"answers": {"q": answer}})
    with pytest.raises(ValueError):
        client.choice("s", "t", {"low": "a", "high": "b"})


@pytest.mark.parametrize("answer", [{"type": "noul", "noul": 1.5}, {"type": "choice", "choice": "x"}, {"type": "noul"}])
def test_invalid_answers_are_rejected(answer):
    client = JevClient(transport=lambda request: {"answers": {"q": answer}})
    with pytest.raises(ValueError):
        client.noul("s", "t")


def test_transient_errors_retry_then_budget_stops(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda seconds: None)
    attempts = []

    def flaky(request):
        attempts.append(1)
        if len(attempts) == 1:
            raise TransientError("529")
        return {"answers": {"q": {"type": "noul", "noul": 0.5}}}
    client = JevClient(transport=flaky, max_calls=2)
    assert client.noul("s", "t") == 0.5 and len(attempts) == 2
    with pytest.raises(RuntimeError, match="budget"):
        client.noul("s", "t")


def test_missing_key_names_the_variable(monkeypatch):
    monkeypatch.delenv("JEV_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="JEV_API_KEY"):
        JevClient()
