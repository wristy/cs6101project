"""Experiment 3 changes exactly one line of ThoughtTracing: the propagation system prompt."""

from pathlib import Path
import subprocess
import sys

import pytest

from experiments.exp3_persona import run, variant
from tests.upstream_source import ROOT, TRACER, differing_statements, method_body

pytestmark = pytest.mark.skipif(not TRACER.exists(), reason="Run scripts/bootstrap.sh")


def test_only_the_system_prompt_differs_from_upstream():
    upstream = method_body(TRACER, "Tracer", "propagate")
    ours = method_body(Path(variant.__file__), "ObserverTracer", "propagate")
    changed = differing_statements(upstream, ours)
    assert len(changed) == 1 and "system_prompt" in upstream[changed[0]]
    assert variant.ORIGINAL.replace("{target_agent}", "") in TRACER.read_text().replace("{target_agent}", "")


def test_plan_runs_one_arm_on_paraphrased_tomi():
    assert run.PLAN == {"mimo": ["paraphrased_tomi"]}
    assert run.ARMS[0]["variant"] == "experiments.exp3_persona.variant"


def test_real_tracer_class_sends_the_observer_prompt(tmp_path):
    code = r'''
from shared.methods.worker import load_upstream, load_variant
upstream = load_upstream("thoughttracing")
tracer = load_variant("experiments.exp3_persona.variant", upstream).__new__(
    load_variant("experiments.exp3_persona.variant", upstream))
calls = []
class Model:
    def batch_interact(self, prompts, system_prompts=None, temperature=0, max_tokens=None):
        calls.append((system_prompts, temperature, max_tokens, prompts))
        return ["new " + str(i) for i in range(len(prompts))]
tracer.tracer_model, tracer.trace_header, tracer.assumption = Model(), "HEADER", ""
tracer.setup_propagation = lambda h, s, p: {"target_agent": "Ava", "context_history": [], "perception_history": [],
                                            "context_and_perception_str": "past", "new_context": "now"}
hypotheses = upstream.HypothesesSetV3("Ava", [], [], ["h1", "h2"], [0.5, 0.5])
result = tracer.propagate(hypotheses, {}, {})
system, temperature, max_tokens, prompts = calls[0]
assert system == "You are a neutral observer watching Ava and trying to predict Ava's thoughts.", system
assert (temperature, max_tokens) == (0, 1024)
assert "<previous prediction regarding Ava's thoughts>\nh1\n" in prompts[0]
assert list(result.texts) == ["new 0", "new 1"]
'''
    result = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout[-3000:] + result.stderr[-3000:]
