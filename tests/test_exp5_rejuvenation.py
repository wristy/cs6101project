"""Experiment 5: only the rejuvenation branch of _trace changes; duplicates are replaced and rescored."""

from pathlib import Path
import subprocess
import sys

import pytest

from experiments.exp5_rejuvenation import run, variant
from tests.upstream_source import ROOT, TRACER, differing_statements, method_body

pytestmark = pytest.mark.skipif(not TRACER.exists(), reason="Run scripts/bootstrap.sh")


def test_trace_loop_differs_from_upstream_only_in_the_rejuvenation_branch():
    upstream = method_body(TRACER, "Tracer", "_trace")
    ours = method_body(Path(variant.__file__), "EmbeddingRejuvenationTracer", "_trace")
    changed = differing_statements(upstream, ours)
    assert len(changed) == 1
    assert "rejuvenate_hypotheses" in upstream[changed[0]] and "replace_duplicates" in ours[changed[0]]
    assert "resample_hypotheses_with_other_info" in ours[changed[0]]   # resampling branch kept


def test_duplicate_slots_drop_the_later_of_each_pair():
    a, a2, b, c = [1, 0, 0], [0.99, 0.05, 0], [0, 1, 0], [0, 0, 1]
    assert variant.duplicate_slots([a, b, a2, c], threshold=0.85) == [2]
    assert variant.duplicate_slots([a, a2, a, b], threshold=0.85) == [1, 2]
    assert variant.duplicate_slots([a, b, c], threshold=0.85) == []


def test_plan():
    assert run.PLAN == {"mimo": ["mmtom"]}


SCRIPT = r'''
from shared import bridge
from shared.models.client import LLMClient, ModelConfig
from shared.models.embeddings import EmbeddingClient
from shared.methods.worker import load_upstream, load_variant

requests = []
def chat(request):
    requests.append(request)
    return {"choices": [{"message": {"content": "1. Ava wants a snack."}, "finish_reason": "stop"}], "usage": {}}
def embed(request):
    table = {"Ava wants food.": [1, 0, 0], "Ava is hungry for food.": [0.99, 0.05, 0],
             "Ava is tired.": [0, 1, 0], "Ava is bored.": [0, 0, 1]}
    return {"data": [{"index": i, "embedding": table[t]} for i, t in enumerate(request["input"])], "usage": {}}
bridge.configure(LLMClient(ModelConfig(), transport=chat))
bridge._extras["embed"] = EmbeddingClient(transport=embed)
upstream = load_upstream("thoughttracing")
cls = load_variant("experiments.exp5_rejuvenation.variant", upstream)
tracer = cls.__new__(cls)
tracer.target_agent = "Ava"
tracer.tracer_model = __import__("shared.bridge", fromlist=["SharedAgent"]).SharedAgent(None)
tracer.interleave_context_and_perception = lambda c, p: "HISTORY"
rescored = []
def weigh(h, action, mode):
    rescored.append(list(h.texts))
    return {"weights": [0.1, 0.2, 0.3, 0.4]}
tracer.weigh = weigh
texts = ["Ava wants food.", "Ava is tired.", "Ava is hungry for food.", "Ava is bored."]
h = upstream.HypothesesSetV3("Ava", [], [], texts, [0.25] * 4)
out = tracer.replace_duplicates(h, "Ava opens the fridge.")
assert out.texts == ["Ava wants food.", "Ava is tired.", "Ava wants a snack.", "Ava is bored."], out.texts
assert [x.text for x in out.hypotheses] == out.texts
assert list(out.weights) == [0.1, 0.2, 0.3, 0.4] and rescored == [out.texts]
assert requests[0]["temperature"] == 0
assert "Ava wants food." in requests[0]["messages"][-1]["content"]
assert "Ava is hungry for food." not in requests[0]["messages"][-1]["content"]
assert tracer.rejuvenation_log[0]["replaced"] == [2]
assert bridge.extra_usage()["embed_calls"] == 1
'''


def test_duplicates_are_replaced_at_temperature_zero_and_all_weights_rescored():
    result = subprocess.run([sys.executable, "-c", SCRIPT], cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout[-3000:] + result.stderr[-3000:]
