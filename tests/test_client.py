"""Step 2: model connections and the agreed A/B probability rule (offline, scripted replies)."""

import math

import pytest

from shared.models.client import LLMClient, ModelConfig, ProtocolError
from shared.models.embeddings import EmbeddingClient, cosine
from shared.models.providers import model_config


def response(text="basket", token=None, logprob=None, finish="stop", top=None):
    content = [{"token": token, "logprob": logprob, "top_logprobs": top or []}] if token is not None else []
    return {"choices": [{"message": {"content": text}, "finish_reason": finish, "logprobs": {"content": content}}],
            "usage": {"prompt_tokens": 12, "completion_tokens": 1}}


def capture(reply):
    requests = []
    return requests, (lambda request: requests.append(request) or reply)


def test_mimo_generation_request_turns_thinking_off_and_logs(tmp_path):
    requests, transport = capture(response())
    client = LLMClient(model_config("mimo"), transport=transport, log_path=tmp_path / "calls.jsonl")
    assert client.generate("story", max_tokens=50) == "basket"
    request = requests[0]
    assert request["model"] == "XiaomiMiMo/MiMo-V2.6-Pro"
    assert request["extra_body"]["chat_template_kwargs"]["enable_thinking"] is False
    assert request["max_tokens"] == 50 and request["temperature"] == 0 and request["seed"] == 42
    assert "reasoning_effort" not in request and "logit_bias" not in request
    assert client.usage()["prompt_tokens"] == 12
    assert (tmp_path / "calls.jsonl").read_text().count("\n") == 1


def test_gpt6sol_generation_request_uses_reasoning_none_and_completion_tokens():
    requests, transport = capture(response())
    client = LLMClient(model_config("gpt6sol"), transport=transport)
    client.generate("story", max_tokens=50, temperature=1)
    request = requests[0]
    assert request["model"] == "gpt-6-sol" and request["reasoning_effort"] == "none"
    assert request["max_completion_tokens"] == 50 and "max_tokens" not in request
    assert request["temperature"] == 1 and "extra_body" not in request


@pytest.mark.parametrize("name,tokens_key,tokens", [("mimo", "max_tokens", 1), ("gpt6sol", "max_completion_tokens", 16)])
def test_scoring_request_matches_autotom(name, tokens_key, tokens):
    requests, transport = capture(response("A", "A", math.log(0.7)))
    LLMClient(model_config(name), transport=transport).score_label("A) Likely. B) Unlikely.")
    request = requests[0]
    assert request[tokens_key] == tokens and request["temperature"] == 0
    assert request["logprobs"] is True and request["top_logprobs"] == 5
    assert request["messages"] == [{"role": "system", "content": "A) Likely. B) Unlikely."}]
    assert "logit_bias" not in request


@pytest.mark.parametrize("token,p,expected", [("A", 0.7, 0.7), ("B", 0.8, 0.2), (" A", 0.6, 0.6), ("B", 1.0, 0.0)])
def test_ab_rule_reads_the_written_letter(token, p, expected):
    client = LLMClient(ModelConfig(), transport=lambda request: response(token.strip(), token, math.log(p)))
    assert client.score_label("prompt") == pytest.approx(expected)


def test_ab_rule_rejects_other_answers_and_missing_logprobs():
    client = LLMClient(ModelConfig(), transport=lambda request: response("Paris", "Paris", -0.1))
    with pytest.raises(ProtocolError, match="A or B"):
        client.score_label("prompt")
    client = LLMClient(ModelConfig(), transport=lambda request: response("A"))
    with pytest.raises(ProtocolError, match="logprob"):
        client.score_label("prompt")


@pytest.mark.parametrize("reply", [response("partial", finish="length"), response("<think>hidden</think>basket")])
def test_rejects_truncation_and_unexpected_thinking(reply):
    client = LLMClient(ModelConfig(), transport=lambda request: reply)
    with pytest.raises(ProtocolError):
        client.generate("prompt")


def test_call_budget_stops_before_transport():
    client = LLMClient(ModelConfig(max_calls=1), transport=lambda request: response())
    client.generate("first")
    with pytest.raises(RuntimeError, match="budget"):
        client.generate("second")


def test_missing_key_names_the_variable(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        LLMClient(model_config("gpt6sol"))


def test_unknown_model_name_is_rejected():
    with pytest.raises(KeyError, match="mimo"):
        model_config("sonnet")


def test_embeddings_keep_input_order_and_cosine():
    reply = {"data": [{"index": 1, "embedding": [0.0, 1.0]}, {"index": 0, "embedding": [1.0, 0.0]}], "usage": {}}
    requests, transport = capture(reply)
    vectors = EmbeddingClient(transport=transport).embed(["first", "second"])
    assert vectors == [[1.0, 0.0], [0.0, 1.0]]
    assert requests[0]["model"] == "Qwen/Qwen3-Embedding-8B"
    assert cosine([1, 0], [1, 0]) == pytest.approx(1.0) and cosine([1, 0], [0, 1]) == pytest.approx(0.0)


def scoring_reply(token, logprob, alternatives=()):
    top = [{"token": t, "logprob": lp} for t, lp in alternatives]
    return {"choices": [{"message": {"content": token}, "finish_reason": "length",
                         "logprobs": {"content": [{"token": token, "logprob": logprob, "top_logprobs": top}]}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1}}


def test_non_letter_answer_uses_autotoms_fallbacks(tmp_path):
    import math
    log = tmp_path / "calls.jsonl"
    client = LLMClient(ModelConfig(), transport=lambda r: scoring_reply("The", math.log(0.9)), log_path=log)
    with pytest.raises(ProtocolError, match="A or B"):
        client.score_label("prompt")
    assert client.score_label("prompt", fallback=0.1) == 0.1
    alternatives = LLMClient(ModelConfig(), transport=lambda r: scoring_reply(
        "The", math.log(0.6), [("The", math.log(0.6)), (" A", math.log(0.3))]))
    assert abs(alternatives.score_label("prompt", fallback=0.1) - 0.3) < 1e-9
    assert client.usage()["score_fallbacks"] == 1 and alternatives.usage()["score_fallbacks"] == 1
    assert '"score_fallback": "constant"' in log.read_text()
