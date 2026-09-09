from dataclasses import replace
import json

import pytest

from tom_research.data import Example
from tom_research.evaluation import run_pilot, score_answer, summarize


def example():
    return Example("test:1", "story", "Ava put the apple in the basket.",
                   "Where will Ava look for the apple?", ("basket", "box"),
                   "Ava", "basket", "first_order_0_no_tom", "true_belief")


def test_free_text_scoring_matches_released_case_insensitive_rule():
    assert score_answer("In the BASKET in the kitchen.", "basket")
    assert not score_answer("In the box.", "basket")
    assert not score_answer("", "basket")
    assert not score_answer(None, "basket")


def test_choice_scoring_does_not_accept_mentions_of_both_options(tmp_path):
    item = replace(example(), answer_format="multiple_choice")
    runner = lambda *args: {"status": "ok", "answer": "basket or box"}
    record = run_pilot([item], ["plain"], runner, tmp_path, {"track": "autotom"})[0]
    assert not record["correct"]
    assert record["status"] == "invalid"
    assert record["track"] == "autotom"


def test_track_rejects_wrong_dataset_format_and_method(tmp_path):
    with pytest.raises(ValueError, match="track"):
        run_pilot([example()], ["plain", "autotom"], None, tmp_path, {"track": "autotom"})
    with pytest.raises(ValueError, match="track"):
        run_pilot([example()], ["autotom"], None, tmp_path, {"track": "thoughttracing"})


def test_summary_keeps_plain_baselines_separate(tmp_path):
    runner = lambda *args: {"status": "ok", "answer": "basket"}
    free_text = run_pilot([example()], ["plain"], runner, tmp_path / "tt", {"track": "thoughttracing"})
    choice = run_pilot([replace(example(), answer_format="multiple_choice")], ["plain"],
                       runner, tmp_path / "auto", {"track": "autotom"})
    reports = summarize(free_text + choice)
    assert len(reports) == 2
    assert {report["track"] for report in reports} == {"thoughttracing", "autotom"}


def test_resume_reuses_records_and_payload_has_no_gold(tmp_path):
    seen = []

    def runner(method, payload, directory):
        assert "answer" not in payload
        seen.append((method, payload))
        return {"answer": "basket", "usage": {"calls": 2}, "status": "ok"}

    config = {"mode": "offline", "seed": 42}
    first = run_pilot([example()], ["plain", "autotom"], runner, tmp_path, config)
    assert len(seen) == 2
    assert first == run_pilot([example()], ["plain", "autotom"], runner, tmp_path, config)
    assert len(seen) == 2
    with pytest.raises(ValueError, match="manifest"):
        run_pilot([example()], ["plain", "autotom"], runner, tmp_path, {**config, "seed": 43})


def test_errors_remain_in_results_and_summary(tmp_path):
    def broken(method, payload, directory):
        raise RuntimeError("backend unavailable")

    records = run_pilot([example()], ["autotom"], broken, tmp_path, {"mode": "offline"})
    report = summarize(records)[0]
    assert report["total"] == 1
    assert report["errors"] == 1
    assert report["correct"] == 0
    assert report["accuracy"] == 0
    assert records[0]["status"] == "error"


def test_duplicate_ids_rejected(tmp_path):
    with pytest.raises(ValueError, match="Duplicate"):
        run_pilot([example(), example()], ["plain"], None, tmp_path, {})


def test_refuses_corrupt_result_file(tmp_path):
    runner = lambda method, payload, directory: {"answer": "basket", "status": "ok"}
    run_pilot([example()], ["plain"], runner, tmp_path, {})
    with (tmp_path / "predictions.jsonl").open("a") as stream:
        stream.write("{broken")
    with pytest.raises(ValueError, match="results"):
        run_pilot([example()], ["plain"], runner, tmp_path, {})