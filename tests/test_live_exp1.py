"""Experiment 1 smoke run: the first frozen question, every arm, real API calls. `pytest --live`.

Plain prompting makes 1 call per question; ThoughtTracing and AutoToM make tens. With MiMo on
four benchmarks and GPT-6 Sol on BigToM this is roughly 300-600 calls in total, a few cents.
Results go to results/live-tests/ (git-ignored). The test checks that each arm completes and
returns a scored answer, not that it is right.
"""

from pathlib import Path
import shutil

import pytest

from experiments.exp1_baseline import run as exp1
from shared.benchmarks import exploretom
from shared.runner import experiment

pytestmark = pytest.mark.live
ROOT = Path(__file__).resolve().parents[1] / "results/live-tests"
KEYS = {"mimo": "DEEPINFRA_API_KEY", "gpt6sol": "OPENAI_API_KEY"}
CASES = [(model, benchmark, arm["name"]) for model, benchmarks in exp1.PLAN.items()
         for benchmark in benchmarks for arm in exp1.ARMS]


@pytest.mark.parametrize("model,benchmark,arm_name", CASES)
def test_one_question_runs_end_to_end(model, benchmark, arm_name, require_key):
    require_key(KEYS[model])
    if benchmark == "exploretom" and not exploretom.SOURCE.exists():
        pytest.skip("ExploreToM file not downloaded")
    arm = next(arm for arm in exp1.ARMS if arm["name"] == arm_name)
    # A smoke test always makes fresh calls: an old error record would otherwise be resumed.
    shutil.rmtree(experiment.result_dir(exp1.EXPERIMENT, model, benchmark, arm_name, 1, ROOT), ignore_errors=True)
    row = experiment.run(exp1.EXPERIMENT, arm, model, benchmark, limit=1, root=ROOT)[0]
    record = experiment.result_dir(exp1.EXPERIMENT, model, benchmark, arm_name, 1, ROOT) / "predictions.jsonl"
    assert row["errors"] == 0, record.read_text()
    assert row["invalid"] == 0, record.read_text()
    assert row["calls"] >= 1
