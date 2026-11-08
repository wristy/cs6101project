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


def test_workers_answer_questions_at_once_and_resume_the_same_way(tmp_path):
    import threading
    import time

    active, peak, lock = [0], [0], threading.Lock()

    def runner(method, payload, directory):
        with lock:
            active[0] += 1
            peak[0] = max(peak[0], active[0])
        time.sleep(0.05)
        with lock:
            active[0] -= 1
        return {"status": "ok", "answer": "basket", "usage": {"calls": 1}}
    items = [example(id=f"test:{i}") for i in range(8)]
    records = run_arm(items[:6], ARM, runner, tmp_path / "a", workers=4)
    assert peak[0] > 1 and sorted(r["id"] for r in records) == sorted(i.id for i in items[:6])
    lines = (tmp_path / "a" / "predictions.jsonl").read_text().splitlines()
    assert len(lines) == 6 and all(json.loads(line)["correct"] for line in lines)
    serial = run_arm(items[:6], ARM, runner, tmp_path / "b", workers=1)
    parallel_row, serial_row = summarize(records)[0], summarize(serial)[0]
    assert {k: parallel_row[k] for k in ("total", "correct", "calls")} == \
        {k: serial_row[k] for k in ("total", "correct", "calls")}


def test_session_limit_stops_new_questions_with_workers(tmp_path):
    records = run_arm([example(id=f"test:{i}") for i in range(4)], ARM,
                      lambda *a: {"status": "ok", "answer": "basket"}, tmp_path, session_seconds=0, workers=2)
    assert records == []
