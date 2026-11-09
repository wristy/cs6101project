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


def test_retry_errors_reruns_only_crashed_questions(tmp_path):
    import json
    from hashlib import sha256
    from shared.runner.evaluation import drop_errors, run_arm
    from shared.benchmarks.common import Example

    examples = [Example(id=f"q{i}", benchmark="paraphrased_tomi", context="c", question="Where?",
                        choices=("box", "basket"), answer="box", target="A", category="first_order")
                for i in range(3)]
    calls = []

    def runner(method, payload, directory):
        calls.append(payload["id"])
        if len(calls) <= 3 and payload["id"] == "q1":
            return {"status": "error", "error": "IndexError: boom"}
        return {"status": "ok", "answer": "the box"}

    arm = {"name": "plain", "method": "plain", "model": "m"}
    run_arm(examples, arm, runner, tmp_path)
    item = tmp_path / "items" / sha256(b"q1").hexdigest()[:16]
    (item / "stale-cache.csv").write_text("from the failed attempt")
    run_arm(examples, arm, runner, tmp_path)                    # plain resume: nothing reruns
    assert calls == ["q0", "q1", "q2"]
    records = run_arm(examples, arm, runner, tmp_path, retry_errors=True)
    assert calls[3:] == ["q1"] and not (item / "stale-cache.csv").exists()
    assert sorted((r["id"], r["status"]) for r in records) == [("q0", "ok"), ("q1", "ok"), ("q2", "ok")]
    retried = [json.loads(line) for line in (tmp_path / "retried.jsonl").read_text().splitlines()]
    assert [r["id"] for r in retried] == ["q1"]
    assert drop_errors(tmp_path) == 0


def test_timed_out_question_keeps_its_token_usage(tmp_path):
    import json
    from shared.methods.runner import usage_from_log
    log = tmp_path / "calls.jsonl"
    lines = [{"status": "ok", "seconds": 2.0, "response": str({"usage": {"prompt_tokens": 100, "completion_tokens": 7}})},
             {"status": "ok", "seconds": 1.0, "response": {"usage": {"prompt_tokens": 50, "completion_tokens": 3}}},
             {"status": "error", "seconds": 9.0}]
    log.write_text("".join(json.dumps(line) + "\n" for line in lines))
    usage = usage_from_log(log)
    assert (usage["calls"], usage["prompt_tokens"], usage["completion_tokens"], usage["seconds"]) == (2, 150, 10, 3.0)
    assert usage_from_log(tmp_path / "missing.jsonl")["calls"] == 0
