"""The shared experiment plumbing and Experiment 1's plan, with a scripted runner."""

import json

import pytest

from experiments.exp1_baseline import run as exp1
from shared.benchmarks import bigtom
from shared.runner import experiment


def test_exp1_plan_covers_every_cell():
    assert {arm["method"] for arm in exp1.ARMS} == {"plain", "autotom", "thoughttracing"}
    assert exp1.PLAN == {"mimo": list(experiment.BENCHMARKS), "gpt6sol": ["bigtom"]}


def test_pilot_results_go_to_their_own_folder():
    full = experiment.result_dir("exp1_baseline", "mimo", "bigtom", "plain")
    pilot = experiment.result_dir("exp1_baseline", "mimo", "bigtom", "plain", limit=2)
    assert full != pilot and "pilot-2" in str(pilot)


@pytest.mark.skipif(not bigtom.SOURCE.exists(), reason="Run scripts/bootstrap.sh")
def test_run_writes_records_and_summary(tmp_path):
    calls = []

    def runner(method, payload, directory):
        calls.append(payload["id"])
        return {"status": "ok", "answer": "a)", "usage": {"calls": 1}}
    rows = experiment.run("exp1_baseline", exp1.ARMS[0], "mimo", "bigtom", limit=2, runner=runner, root=tmp_path)
    folder = tmp_path / "exp1_baseline/mimo/pilot-2/bigtom/plain"
    assert len(calls) == 2 and rows[0]["total"] == 2
    assert json.loads((folder / "summary.json").read_text()) == rows
    assert json.loads((folder / "manifest.json").read_text())["arm"]["model_config"] == "XiaomiMiMo/MiMo-V2.6-Pro"
