"""Experiment 6: one cached running summary feeds propagation and weighting; the answer keeps the full trace."""

import subprocess
import sys

import pytest

from experiments.exp6_summary import run
from tests.upstream_source import ROOT, TRACER

pytestmark = pytest.mark.skipif(not TRACER.exists(), reason="Run scripts/bootstrap.sh")


def test_plan():
    assert run.PLAN == {"mimo": ["mmtom"]}


SCRIPT = r'''
from shared import bridge
from shared.models.client import LLMClient, ModelConfig
from shared.methods.worker import load_upstream, load_variant

requests = []
def chat(request):
    requests.append(request)
    return {"choices": [{"message": {"content": f"S{len(requests)}"}, "finish_reason": "stop"}], "usage": {}}
bridge.configure(LLMClient(ModelConfig(), transport=chat))
upstream = load_upstream("thoughttracing")
cls = load_variant("experiments.exp6_summary.variant", upstream)
tracer = cls.__new__(cls)
tracer.target_agent = "Ava"
tracer.assumption = "\nIf Ava has been trying to get an Object H, SECRET"
tracer.args = type("A", (), {"input_is_chat": False})()
contexts = [{"state": "Room C has a fridge.", "action": None},
            {"state": None, "action": "Ava walks to Room C."},
            {"state": None, "action": "Ava opens the fridge."}]
perceptions = [{"state": "Ava sees the fridge.", "action": None},
               {"state": None, "action": "Ava sees Room C."},
               {"state": None, "action": "Ava sees inside the fridge."}]

# Full history outside weighting, as upstream (used for the final answer).
full = tracer.interleave_context_and_perception(contexts, perceptions)
assert "Ava walks to Room C." in full and "<summary" not in full

# Propagation: summary of the earlier steps replaces the raw history; the new step is upstream's.
h = upstream.HypothesesSetV3("Ava", contexts[:2], perceptions[:2], ["t"], [1.0])
info = tracer.setup_propagation(h, contexts[2], perceptions[2])
assert info["context_and_perception_str"].count("S2") == 1 and "walks" not in info["context_and_perception_str"]
assert "Ava opens the fridge." in info["new_context"]
assert len(requests) == 2

# Weighting: summary of earlier steps plus the current step; cached, so no new summary calls.
tracer._weighing = True
text = tracer.interleave_context_and_perception(contexts[:2], perceptions[:2])
tracer._weighing = False
assert text.startswith("<summary of what Ava has perceived so far>\nS1") and "Ava walks to Room C." in text
assert len(requests) == 2
# The summary never sees the question or the assumption sentence.
assert not any("SECRET" in m["content"] for r in requests for m in r["messages"])
assert requests[1]["messages"][-1]["content"].startswith("<previous summary>\nS1")
'''


def test_summary_feeds_propagation_and_weighting_only():
    result = subprocess.run([sys.executable, "-c", SCRIPT], cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout[-3000:] + result.stderr[-3000:]
