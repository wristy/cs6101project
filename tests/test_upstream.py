from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]


def run_script(code, tmp_path):
    if not (ROOT / "external/AutoToM/model/ProbSolver.py").exists():
        pytest.skip("Fetch pinned upstream sources with scripts/bootstrap.sh")
    result = subprocess.run([sys.executable, "-c", code, str(tmp_path)], cwd=ROOT,
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr


def test_real_thoughttracing_one_step_with_scripted_responses(tmp_path):
    run_script('''
import json
from pathlib import Path
import sys
from shared.models.client import LLMClient, ModelConfig
from shared.methods.worker import run_method

outputs = iter([
    "Ava is in the kitchen. The apple is in the basket.<no action>",
    "Ava sees the apple in the basket.",
    "Ava is in the kitchen.",
    "1. The apple is in the basket.\\n2. The kitchen is familiar.\\n3. Nobody moved anything.\\n4. The basket holds food.",
    "Ava believes the apple is in the basket.",
    "In the basket in the kitchen.",
])
def transport(request):
    return {"choices": [{"message": {"content": next(outputs)}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 10}}
client = LLMClient(ModelConfig(), transport=transport)
payload = {"context": "Ava is in the kitchen. The apple is in the basket.",
           "question": "Where will Ava look for the apple?", "target": "Ava", "choices": ["basket"]}
result = run_method("thoughttracing", payload, client, Path(sys.argv[1]))
assert result["status"] == "ok"
assert result["answer"] == "In the basket in the kitchen."
assert client.usage()["calls"] == 6
saved = json.loads(next(Path(sys.argv[1]).glob("tracer-*.jsonl")).read_text())
assert len(saved["hypotheses"]) == 1
assert "Where will Ava look" not in str([record["request"] for record in client.records[:-1]])
''', tmp_path)


def test_real_autotom_likelihood_and_generation_use_shared_backend(tmp_path):
    run_script('''
import math
from shared.bridge import configure
from shared.models.client import LLMClient, ModelConfig
from shared.methods.worker import load_upstream

def transport(request):
    return {"choices": [{"message": {"content": "A"}, "finish_reason": "stop",
            "logprobs": {"content": [{"token": "A", "logprob": math.log(0.4), "top_logprobs": [
                {"token": "A", "bytes": [65], "logprob": math.log(0.4)}]}]}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 1}}
client = LLMClient(ModelConfig(), transport=transport)
configure(client)
load_upstream("autotom")
import probs
import utils
probability = probs.get_likelihood("Ava sees a basket", "Ava looks in the basket", "customize",
                                   model=client.config.model, variable="Action", inf_agent="Ava")
assert abs(probability - 0.4) < 1e-8
powered = probs.get_likelihood("Ava sees a basket", "Ava looks in the basket", "customize",
                               model=client.config.model, variable="Action", inf_agent="Ava", action_exponent=2)
assert abs(powered - 0.16) < 1e-8
assert utils.llm_request("Describe the scene", model=client.config.model) == ("A", 0.0)
assert all(record["request"]["model"] == client.config.model for record in client.records)
assert client.usage()["calls"] == 3
''', tmp_path)