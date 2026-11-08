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


def test_score_request_uses_ordered_criteria():
    sent = []

    def transport(request):
        sent.append(request)
        return {"model": "m", "answers": {"q": {"type": "score", "score": 1.2, "confidence": 0.6,
                                                "probabilities": {"0": 0.2, "1": 0.4, "2": 0.4}}}}
    answer = JevClient(transport=transport).score("story", "Which model?", ["low", "mid", "high"])
    assert sent[0]["questions"]["q"] == {"type": "score", "instructions": "Which model?", "criteria": ["low", "mid", "high"]}
    assert answer["probabilities"]["1"] == 0.4


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
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="TYPESAFE_API_KEY"):
        JevClient()
