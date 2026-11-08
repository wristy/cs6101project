"""Experiments 2-6 smoke run: the first frozen question through each variant arm. `pytest --live`.

Each case is about the cost of one ThoughtTracing question (30-50 calls). Results go to
results/live-tests/ (git-ignored). The test checks that the arm completes and is scored.
"""

import importlib
from pathlib import Path
import shutil

import pytest

from shared.runner import experiment

pytestmark = pytest.mark.live
ROOT = Path(__file__).resolve().parents[1] / "results/live-tests"
EXPERIMENTS = ["exp2a_mental_models", "exp2b_triples", "exp3_persona", "exp4_scoring",
               "exp5_rejuvenation", "exp6_summary"]
CASES = [(name, arm["name"], benchmark)
         for name in EXPERIMENTS
         for arm in importlib.import_module(f"experiments.{name}.run").ARMS
         for benchmark in importlib.import_module(f"experiments.{name}.run").PLAN["mimo"]]


@pytest.mark.parametrize("name,arm_name,benchmark", CASES)
def test_variant_runs_one_question(name, arm_name, benchmark, require_key):
    require_key("DEEPINFRA_API_KEY")
    module = importlib.import_module(f"experiments.{name}.run")
    arm = next(arm for arm in module.ARMS if arm["name"] == arm_name)
    if "noul" in arm.get("variant", "") or "jev" in arm_name or name == "exp2a_mental_models":
        require_key("TYPESAFE_API_KEY")
    # A smoke test always makes fresh calls: an old error record would otherwise be resumed.
    shutil.rmtree(experiment.result_dir(module.EXPERIMENT, "mimo", benchmark, arm_name, 1, ROOT), ignore_errors=True)
    row = experiment.run(module.EXPERIMENT, arm, "mimo", benchmark, limit=1, root=ROOT)[0]
    record = experiment.result_dir(module.EXPERIMENT, "mimo", benchmark, arm_name, 1, ROOT) / "predictions.jsonl"
    assert row["errors"] == 0 and row["invalid"] == 0, record.read_text()
