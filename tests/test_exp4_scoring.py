"""Experiment 4: only prompt_likelihood changes; templates match their sources; weights normalise."""

from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from experiments.exp4_scoring import run, variants
from tests.upstream_source import ROOT, TRACER

AUTOTOM_PROBS = ROOT / "external/AutoToM/model/probs.py"
pytestmark = pytest.mark.skipif(not TRACER.exists(), reason="Run scripts/bootstrap.sh")


def test_autotom_template_is_word_for_word():
    upstream = variants.AUTOTOM_ACTION.replace(
        "it's unlikely.\nDetermine", "it's unlikely.{rational_statement}\nDetermine")
    assert f'prompt = f"""{upstream}"""' in AUTOTOM_PROBS.read_text()


def test_normalise_and_spread():
    assert np.allclose(variants.normalise([0.2, 0.2, 0.4, 0.2]), [0.2, 0.2, 0.4, 0.2])
    assert np.allclose(variants.normalise([0, 0, 0, 0]), [0.25] * 4)
    assert variants._results([], [0.1, 0.3])["weight_spread"] == pytest.approx(0.5)


def test_plan():
    assert run.PLAN == {"mimo": ["paraphrased_tomi"]}
    assert {arm["variant"] for arm in run.ARMS} == {"experiments.exp4_scoring.ab", "experiments.exp4_scoring.noul"}


SCRIPT = r'''
import math
from shared import bridge
from shared.models.client import LLMClient, ModelConfig
from shared.models.typesafe import JevClient
from shared.methods.worker import load_upstream, load_variant

def letter(request):
    token = "A" if "food" in request["messages"][-1]["content"] else "B"
    return {"choices": [{"message": {"content": token}, "finish_reason": "length",
            "logprobs": {"content": [{"token": token, "logprob": math.log(0.8)}]}}], "usage": {}}
jev_sent = []
def jev(request):
    jev_sent.append(request)
    p = 0.9 if "food" in request["questions"]["q"]["instructions"] else 0.3
    return {"model": "jev-1.13.0", "answers": {"q": {"type": "noul", "noul": p}}, "usage": {"input_tokens": 5}}
client = LLMClient(ModelConfig(), transport=letter)
bridge.configure(client)
bridge._extras["jev"] = JevClient(transport=jev)
upstream = load_upstream("thoughttracing")
contexts = [{"state": "Ava is home.", "action": "Ava walks to the kitchen."}]
perceptions = [{"state": "Ava sees home.", "action": "Ava sees the kitchen."}]
for spec in ("experiments.exp4_scoring.ab", "experiments.exp4_scoring.noul"):
    cls = load_variant(spec, upstream)
    tracer = cls.__new__(cls)
    tracer.target_agent = "Ava"
    result = tracer.prompt_likelihood(["Ava wants food from the kitchen.", "Ava wants to sleep."],
                                      contexts, perceptions, "Ava walks to the kitchen.")
    weights = result["weights"]
    assert abs(sum(weights) - 1) < 1e-9 and weights[0] > weights[1], (spec, weights)
    # Upstream's resampling copies weight_details["reasonings"]; it crashed 7 full-run questions.
    texts = ["Ava wants food from the kitchen.", "Ava wants to sleep."]
    hypotheses = upstream.HypothesesSetV3("Ava", contexts, perceptions, texts, weights)
    hypotheses.weight_details = result
    resampled = upstream.resample_hypotheses_with_other_info(hypotheses, 1.0)
    assert len(resampled.texts) == 2 and len(resampled.weight_details["reasonings"]) == 2
    if spec.endswith("ab"):
        assert list(result["raw_scores"]) == [0.8, 1 - 0.8]
        prompt = client.records[0]["request"]["messages"][-1]["content"]
        assert "State: Ava is home.\nAva's Belief: Ava wants food from the kitchen." in prompt
        assert "Determine if the following statement is likely: Ava walks to the kitchen." in prompt
    else:
        assert list(result["raw_scores"]) == [0.9, 0.3]
        assert jev_sent[0]["state"] == "Ava is home."
        assert jev_sent[0]["questions"]["q"]["instructions"] == (
            "Given the context and Ava's thoughts [Ava wants food from the kitchen.], "
            "Ava would do the next action: Ava walks to the kitchen.")
assert bridge.extra_usage()["jev_calls"] == 2
'''


def test_both_variants_weight_real_tracer_hypotheses():
    result = subprocess.run([sys.executable, "-c", SCRIPT], cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout[-3000:] + result.stderr[-3000:]
