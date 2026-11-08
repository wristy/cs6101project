"""The arm runner: resumable records, manifest guard, errors kept, summaries per arm."""

import json

import pytest

from shared.benchmarks.common import Example
from shared.runner.evaluation import run_arm, summarize

ARM = {"name": "plain", "method": "plain"}


def example(id="test:1", answer="basket", category="first_order_0_no_tom"):
    return Example(id=id, benchmark="paraphrased_tomi", context="Ava put the apple in the basket.",
                   question="Where will Ava look for the apple?", choices=("basket", "box"),
                   target="Ava", answer=answer, category=category, belief_type="true_belief",
                   answer_format="free_text", native={"thoughttracing": {}, "autotom": {}})


def test_records_are_scored_and_payload_has_no_gold(tmp_path):
    def runner(method, payload, directory):
        assert "answer" not in payload and method == "plain"
        return {"status": "ok", "answer": "In the basket.", "usage": {"calls": 1}}
    record = run_arm([example()], ARM, runner, tmp_path)[0]
    assert record["correct"] and record["gold"] == "basket" and record["arm"] == "plain"
    assert record["benchmark"] == "paraphrased_tomi" and record["category"] == "first_order_0_no_tom"


def test_resume_skips_finished_questions_and_guards_manifest(tmp_path):
    seen = []

    def runner(method, payload, directory):
        seen.append(payload["id"])
        return {"status": "ok", "answer": "basket"}
    items = [example("test:1"), example("test:2")]
    first = run_arm(items, ARM, runner, tmp_path)
    assert run_arm(items, ARM, runner, tmp_path) == first and seen == ["test:1", "test:2"]
    with pytest.raises(ValueError, match="manifest"):
        run_arm(items, {**ARM, "name": "other"}, runner, tmp_path)
    with pytest.raises(ValueError, match="manifest"):
        run_arm(items[:1], ARM, runner, tmp_path)


def test_errors_stay_in_results_and_summary(tmp_path):
    def broken(method, payload, directory):
        raise RuntimeError("backend unavailable")
    records = run_arm([example()], {"name": "tt", "method": "thoughttracing"}, broken, tmp_path)
    row = summarize(records)[0]
    assert records[0]["status"] == "error" and "backend unavailable" in records[0]["error"]
    assert (row["total"], row["errors"], row["correct"], row["accuracy"]) == (1, 1, 0, 0)


def test_summary_splits_by_belief_and_category(tmp_path):
    runner = lambda method, payload, directory: {"status": "ok", "answer": "basket", "usage": {"calls": 2}}
    items = [example("test:1"), example("test:2", answer="box", category="second_order_0_no_tom")]
    row = summarize(run_arm(items, ARM, runner, tmp_path))[0]
    assert row["accuracy"] == 0.5 and row["calls"] == 4
    assert row["accuracy[first_order_0_no_tom]"] == 1 and row["accuracy[second_order_0_no_tom]"] == 0


def test_rejects_duplicates_unknown_methods_and_variants_outside_tracing(tmp_path):
    with pytest.raises(ValueError, match="Duplicate"):
        run_arm([example(), example()], ARM, None, tmp_path)
    with pytest.raises(ValueError, match="Unknown method"):
        run_arm([example()], {"name": "x", "method": "cot"}, None, tmp_path)
    with pytest.raises(ValueError, match="Variants"):
        run_arm([example()], {"name": "x", "method": "plain", "variant": "v"}, None, tmp_path)


def test_refuses_corrupt_result_file(tmp_path):
    runner = lambda method, payload, directory: {"answer": "basket", "status": "ok"}
    run_arm([example()], ARM, runner, tmp_path)
    with (tmp_path / "predictions.jsonl").open("a") as stream:
        stream.write("{broken")
    with pytest.raises(ValueError, match="Corrupt"):
        run_arm([example()], ARM, runner, tmp_path)
