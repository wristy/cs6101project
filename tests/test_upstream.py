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
    "Target character: Ava",
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
context = "Ava is in the kitchen. The apple is in the basket."
question = "Where will Ava look for the apple?"
payload = {"id": "t:0", "context": context, "question": question, "target": "Ava", "choices": ["basket"],
           "native": {"thoughttracing": {"dataset": "tomi",
                                         "input": f"{context}\\n\\nQuestion: {question}\\nAnswer:"}}}
result = run_method("thoughttracing", payload, client, Path(sys.argv[1]))
assert result["status"] == "ok"
assert result["answer"] == "In the basket in the kitchen."
assert client.usage()["calls"] == 7
saved = json.loads(next(Path(sys.argv[1]).glob("tracer-*.jsonl")).read_text())
assert len(saved["hypotheses"]) == 1
# Only target identification and the final answer see the question, as upstream.
assert "Where will Ava look" in str(client.records[0]["request"])
assert "Where will Ava look" not in str([record["request"] for record in client.records[1:-1]])
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

def test_autotom_question_classifiers_use_shared_backend_not_gpt4o(tmp_path):
    """model_adjustment.gpt_call (nested/realistic/memory checks, model proposal) must use our model."""
    run_script('''
import os
os.environ["OPENAI_API_KEY"] = "sk-must-not-be-used"
from shared.bridge import configure
from shared.models.client import LLMClient, ModelConfig
from shared.methods.worker import load_upstream

def transport(request):
    return {"choices": [{"message": {"content": "No"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 1}}
client = LLMClient(ModelConfig(), transport=transport)
configure(client)
load_upstream("autotom")
import model_adjustment
assert model_adjustment.determine_higher_order_belief("Where will Ava look for the apple?") is False
assert client.usage()["calls"] == 1 and client.records[0]["request"]["model"] == client.config.model
os.environ["TOM_RESEARCH_BACKEND"] = "0"
try:
    model_adjustment.gpt_call("hi")
except RuntimeError as error:
    assert "unpatched AutoToM" in str(error)
else:
    raise AssertionError("direct OpenAI call was not blocked")
''', tmp_path)


def test_worker_drops_openai_key_after_building_its_client(tmp_path, monkeypatch):
    import json
    request = {"method": "plain", "payload": {"native": {"thoughttracing": {"dataset": "tomi", "input": "x"}}},
               "model_config": {"seed": 0}}
    (tmp_path / "request.json").write_text(json.dumps(request))
    code = f'''
import os, sys
os.environ["OPENAI_API_KEY"] = "sk-test"
os.environ["DEEPINFRA_API_KEY"] = "di-test"
import shared.methods.worker as worker
seen = {{}}
def fake_run(*args, **kwargs):
    seen["key"] = os.environ.get("OPENAI_API_KEY")
    return {{"status": "ok"}}
worker.run_method = fake_run
sys.argv = ["worker", "--request", "{tmp_path / 'request.json'}", "--result", "{tmp_path / 'result.json'}"]
worker.main()
assert seen["key"] is None, seen
'''
    run_script(code, tmp_path)


def test_autotom_finds_every_prompt_it_reads_from_our_runtime_folder(tmp_path):
    """verify_variable reads prompts/prompts_gpt-4o/ by name. If it is missing, AutoToM swallows
    the error and drops every extracted action and goal on BigToM (found in the pilot)."""
    run_script('''
import os, sys
from pathlib import Path
from shared.bridge import configure
from shared.models.client import LLMClient, ModelConfig
from shared.methods.worker import load_upstream, prepare_autotom_runtime

def transport(request):
    return {"choices": [{"message": {"content": "A"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1}}
client = LLMClient(ModelConfig(), transport=transport)
configure(client)
runtime = Path(sys.argv[1]) / "model"
runtime.mkdir()
prepare_autotom_runtime(runtime, client.config.model)
os.chdir(runtime)
load_upstream("autotom")
import ElementExtractor
assert ElementExtractor.verify_variable("Action", ["A", "Noor makes the cappuccino."]) == "A"
assert client.usage()["calls"] == 1
''', tmp_path)
