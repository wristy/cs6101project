import math

import pytest

from tom_research.client import LLMClient, ModelConfig, ProtocolError


def response(text="basket", top=None, finish="stop"):
    return {
        "choices": [{"message": {"content": text}, "finish_reason": finish,
                     "logprobs": {"content": [{"top_logprobs": top or []}]}}],
        "usage": {"prompt_tokens": 12, "completion_tokens": 1},
    }


def test_generation_routes_model_and_disables_thinking(tmp_path):
    requests = []
    client = LLMClient(ModelConfig(), transport=lambda request: requests.append(request) or response(),
                       log_path=tmp_path / "calls.jsonl")
    assert client.generate("story") == "basket"
    assert requests[0]["model"] == "Qwen/Qwen3.8-27B"
    assert requests[0]["extra_body"]["chat_template_kwargs"]["enable_thinking"] is False
    assert client.usage()["prompt_tokens"] == 12
    assert client.usage()["calls"] == 1
    assert (tmp_path / "calls.jsonl").read_text().count("\n") == 1


def test_label_score_is_raw_probability_not_ab_normalization():
    top = [{"token": "A", "bytes": [65], "logprob": math.log(0.4)},
           {"token": "B", "bytes": [66], "logprob": math.log(0.1)}]
    requests = []
    client = LLMClient(ModelConfig(), transport=lambda request: requests.append(request) or response("A", top))
    assert client.score_label("A) Likely. B) Unlikely.") == pytest.approx(0.4)
    assert requests[0]["max_tokens"] == 1
    assert requests[0]["top_p"] == 1
    assert "logit_bias" not in requests[0]
    assert requests[0]["messages"] == [{"role": "system", "content": "A) Likely. B) Unlikely."}]


@pytest.mark.parametrize("top", [[], [{"token": " A", "bytes": [32, 65], "logprob": -0.1}]])
def test_missing_exact_label_raises(top):
    client = LLMClient(ModelConfig(), transport=lambda request: response("B", top))
    with pytest.raises(ProtocolError, match="label"):
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
    assert client.usage()["calls"] == 1