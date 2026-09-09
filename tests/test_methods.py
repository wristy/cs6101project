from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from tom_research.bridge import SharedAgent, configure
from tom_research.client import LLMClient, ModelConfig
from tom_research.worker import run_method


ROOT = Path(__file__).resolve().parents[1]


def reply(text):
    return {"choices": [{"message": {"content": text}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1}}


def test_thoughttracing_agent_batch_preserves_system_and_temperature():
    requests = []
    client = LLMClient(ModelConfig(), transport=lambda request: requests.append(request) or reply("belief"))
    configure(client)
    agent = SharedAgent(SimpleNamespace(model=client.config.model))
    assert agent.batch_interact(["one", "two"], system_prompts=["first", "second"], temperature=1) == ["belief", "belief"]
    assert requests[1]["messages"][0]["content"] == "second"
    assert requests[0]["temperature"] == 1


@pytest.mark.parametrize("method", ["thoughttracing", "autotom"])
def test_official_algorithm_imports_without_gpu_dependencies(method):
    if not (ROOT / "external/thought-tracing/tracer.py").exists():
        pytest.skip("Run scripts/bootstrap.sh to fetch pinned sources")
    result = subprocess.run(
        [sys.executable, "-m", "tom_research.worker", "--check-import", method],
        cwd=ROOT, capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert '"status": "ok"' in result.stdout


def test_plain_prompt_is_free_text_without_candidates(tmp_path):
    requests = []
    client = LLMClient(ModelConfig(), transport=lambda request: requests.append(request) or reply("basket"))
    payload = {"context": "Ava sees a basket.", "question": "Where will Ava look?",
               "target": "Ava", "choices": ["basket", "secret distractor"]}
    result = run_method("plain", payload, client, tmp_path)
    assert result["answer"] == "basket"
    assert "secret distractor" not in str(requests)


def test_autotom_plain_baseline_receives_same_choices(tmp_path):
    requests = []
    client = LLMClient(ModelConfig(), transport=lambda request: requests.append(request) or reply("basket"))
    payload = {"context": "story", "question": "Where?", "choices": ["basket", "box"],
               "answer_format": "multiple_choice"}
    run_method("plain", payload, client, tmp_path)
    prompt = requests[0]["messages"][-1]["content"]
    assert "basket" in prompt and "box" in prompt
    assert "exactly" in prompt


def test_tracer_receives_only_context_and_target(tmp_path):
    observed = []

    class Tracer:
        def __init__(self, args):
            assert args.n_hypotheses == 4 and not args.use_helper_llm

        def trace(self, context, target_agent):
            observed.append((context, target_agent))
            return "recorded trace"

    client = LLMClient(ModelConfig(), transport=lambda request: reply("basket"))
    payload = {"context": "story", "question": "test question", "target": "Ava", "choices": []}
    result = run_method("thoughttracing", payload, client, tmp_path, upstream=SimpleNamespace(Tracer=Tracer))
    assert observed == [("story", "Ava")]
    assert result["trace"] == "recorded trace"
    assert "story" in client.records[-1]["request"]["messages"][-1]["content"]
    assert "test question" in client.records[-1]["request"]["messages"][-1]["content"]


def test_autotom_retains_automated_solver_and_maps_posterior_to_text(tmp_path):
    class Solver:
        def __init__(self, **kwargs):
            assert kwargs["model_name"] == "automated"
            assert kwargs["reduce_hypotheses"] and kwargs["back_inference"]
            assert not kwargs["no_model_adjustment"]
            assert kwargs["llm"] == kwargs["hypo_llm"] == "Qwen/Qwen3.8-27B"
            assert "answer" not in kwargs

        def solve(self):
            return [0.25, 0.75], {"model": "belief"}

    client = LLMClient(ModelConfig(), transport=lambda request: reply("unused"))
    payload = {"id": "test:0", "context": "story", "question": "question",
               "target": "Ava", "choices": ["box", "basket"]}
    upstream = SimpleNamespace(ProblemSolver=Solver, argmax=lambda values: values.index(max(values)))
    result = run_method("autotom", payload, client, tmp_path, upstream=upstream)
    assert result["answer"] == "basket"
    assert result["posterior"] == [0.25, 0.75]