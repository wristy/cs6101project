from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from shared.bridge import SharedAgent, configure
from shared.models.client import LLMClient, ModelConfig
from shared.methods import prompts
from shared.methods.worker import run_method


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
        [sys.executable, "-m", "shared.methods.worker", "--check-import", method],
        cwd=ROOT, capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert '"status": "ok"' in result.stdout


def tomi_payload(**native):
    tt_input = "Ava sees a basket.\n\nQuestion: Where will Ava look?\nAnswer:"
    return {"id": "t:0", "context": "Ava sees a basket.", "question": "Where will Ava look?",
            "target": "Ava", "choices": ["basket", "secret distractor"],
            "native": {"thoughttracing": {"dataset": "tomi", "input": tt_input},
                       "autotom": {"dataset_name": "ToMi-1st", **native}}}


def test_plain_uses_thoughttracings_prompt_and_never_the_options(tmp_path):
    requests = []
    client = LLMClient(ModelConfig(), transport=lambda request: requests.append(request) or reply("basket"))
    result = run_method("plain", tomi_payload(), client, tmp_path)
    assert result["answer"] == "basket"
    prompt = requests[0]["messages"][-1]["content"]
    assert prompt.endswith(prompts.TOMI_ANSWER) and "secret distractor" not in prompt
    assert requests[0]["max_tokens"] == 256


def test_tracer_gets_the_native_input_and_finds_the_target_itself(tmp_path):
    observed = []

    class Tracer:
        def __init__(self, args):
            assert args.n_hypotheses == 4 and not args.use_helper_llm and args.dataset == "tomi"

        def trace(self, text, target_agent=None):
            observed.append((text, target_agent))
            return "recorded trace"

    client = LLMClient(ModelConfig(), transport=lambda request: reply("basket"))
    result = run_method("thoughttracing", tomi_payload(), client, tmp_path, upstream=SimpleNamespace(Tracer=Tracer))
    assert observed == [(tomi_payload()["native"]["thoughttracing"]["input"], None)]
    final = client.records[-1]["request"]["messages"][-1]["content"]
    assert "recorded trace" in final and final.endswith(prompts.TOMI_THEREFORE)
    assert result["trace"] == "recorded trace"


def test_mmtom_target_comes_from_the_question_only(tmp_path):
    seen = {}

    class Tracer:
        def __init__(self, args):
            pass

        def identify_target(self, text):
            seen["identify"] = text
            return "Ava"

        def trace(self, text, target_agent=None):
            seen["target"] = target_agent
            return "trace"

    payload = tomi_payload()
    payload["native"]["thoughttracing"] = {"dataset": "mmtom", "input":
        "Apartment.\nQuestion: Which is true? (a) x (b) y " + prompts.MMTOM_RESPOND + "\nAnswer: "}
    client = LLMClient(ModelConfig(), transport=lambda request: reply("(a)"))
    run_method("thoughttracing", payload, client, tmp_path, upstream=SimpleNamespace(Tracer=Tracer))
    assert seen == {"identify": "Question: Which is true? (a) x (b) y " + prompts.MMTOM_RESPOND + "\nAnswer: ",
                    "target": "Ava"}


def test_variant_must_subclass_the_upstream_tracer(tmp_path, monkeypatch):
    import types, sys as system
    class Tracer:
        def __init__(self, args):
            pass

        def trace(self, text, target_agent=None):
            return "variant trace"
    module = types.ModuleType("fake_variant")
    module.build = lambda upstream: type("Variant", (upstream.Tracer,), {})
    monkeypatch.setitem(system.modules, "fake_variant", module)
    client = LLMClient(ModelConfig(), transport=lambda request: reply("basket"))
    result = run_method("thoughttracing", tomi_payload(), client, tmp_path,
                        upstream=SimpleNamespace(Tracer=Tracer), variant="fake_variant")
    assert result["trace"] == "variant trace"
    module.build = lambda upstream: object
    with pytest.raises(TypeError, match="subclass"):
        run_method("thoughttracing", tomi_payload(), client, tmp_path,
                   upstream=SimpleNamespace(Tracer=Tracer), variant="fake_variant")


def test_autotom_runs_with_its_own_main_settings(tmp_path):
    class Solver:
        def __init__(self, **kwargs):
            assert kwargs["model_name"] == "automated" and kwargs["nested"] is None
            assert kwargs["reduce_hypotheses"] and kwargs["back_inference"]
            assert not kwargs["no_model_adjustment"]
            assert kwargs["llm"] == kwargs["hypo_llm"] == "XiaomiMiMo/MiMo-V2.6-Pro"
            assert kwargs["dataset_name"] == "ToMi-1st" and kwargs["precomputed_states"] == ["s"]
            assert kwargs["episode_name"] == "t:0"
            assert "answer" not in kwargs

        def solve(self):
            return [0.25, 0.75], {"model": "belief"}

    client = LLMClient(ModelConfig(), transport=lambda request: reply("unused"))
    upstream = SimpleNamespace(ProblemSolver=Solver, argmax=max, argmin=min)
    result = run_method("autotom", tomi_payload(states=["s"], actions=["a"]), client, tmp_path, upstream=upstream)
    assert result["answer"] == "secret distractor"
    assert result["posterior"] == [0.25, 0.75]


def test_autotom_finds_its_prompts_for_any_model_name(tmp_path):
    from shared.methods.worker import ROOT as REPO, prepare_autotom_runtime
    link = prepare_autotom_runtime(tmp_path, "XiaomiMiMo/MiMo-V2.6-Pro")
    assert link == tmp_path / "prompts/prompts_XiaomiMiMo/MiMo-V2.6-Pro"
    assert link.resolve() == (REPO / "external/AutoToM/model/prompts/prompts_gpt-4o").resolve()
    assert prepare_autotom_runtime(tmp_path, "XiaomiMiMo/MiMo-V2.6-Pro") == link


def test_autotom_declining_a_question_is_scored_wrong_not_a_crash(tmp_path):
    class Solver:
        def __init__(self, **kwargs):
            self.nested = None

        def solve(self):
            self.nested = True   # AutoToM's check_nested refused: higher-order on a first-order set
            return None, {}

    client = LLMClient(ModelConfig(), transport=lambda request: reply("unused"))
    upstream = SimpleNamespace(ProblemSolver=Solver, argmax=max, argmin=min)
    result = run_method("autotom", tomi_payload(states=["s"], actions=["a"]), client, tmp_path, upstream=upstream)
    assert result["status"] == "invalid" and result["nested"] is True and "answer" not in result


def test_every_weighting_step_is_logged_unchanged(tmp_path):
    import numpy as np

    class Tracer:
        def __init__(self, args):
            pass

        def weigh(self, hypotheses, action, mode):
            return {"raw_scores": np.array([0.9, 0.1]), "weights": np.array([0.7, 0.3]), "extra": "kept"}

        def trace(self, text, target_agent=None):
            assert self.weigh(None, "act", "prompting")["extra"] == "kept"
            self.weigh(None, "act2", "prompting")
            return "trace"

    client = LLMClient(ModelConfig(), transport=lambda request: reply("basket"))
    result = run_method("thoughttracing", tomi_payload(), client, tmp_path, upstream=SimpleNamespace(Tracer=Tracer))
    assert result["weights_log"] == [{"raw_scores": [0.9, 0.1], "weights": [0.7, 0.3]}] * 2


def test_autotom_ties_go_to_the_first_option_and_are_flagged(tmp_path):
    class Solver:
        def __init__(self, **kwargs):
            pass

        def solve(self):
            return [0.5, 0.5], {}

    client = LLMClient(ModelConfig(), transport=lambda request: reply("unused"))
    upstream = SimpleNamespace(ProblemSolver=Solver, argmax=max, argmin=min)
    payload = tomi_payload(states=["s"], actions=["a"])
    result = run_method("autotom", payload, client, tmp_path, upstream=upstream)
    assert result["tie"] is True and result["answer"] == payload["choices"][0]
