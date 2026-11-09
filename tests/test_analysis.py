"""Paired statistics and the aggregate outputs, on synthetic results folders."""

import csv
import json

import pytest

from shared.analysis import aggregate
from shared.analysis.stats import mcnemar_exact, paired_bootstrap


def test_mcnemar_exact_matches_known_values():
    assert mcnemar_exact([1, 1, 0], [1, 1, 0])["p"] == 1.0
    result = mcnemar_exact([1] * 10 + [0] * 0, [0] * 10)          # 10 discordant, all one way
    assert result["only_control"] == 10 and result["p"] == pytest.approx(2 / 2 ** 10)
    assert mcnemar_exact([1, 0], [0, 1])["p"] == 1.0


def test_paired_bootstrap_interval_contains_the_difference():
    control = [1, 0] * 50
    variant = [1, 1] * 50
    result = paired_bootstrap(control, variant, reps=2000)
    assert result["diff"] == pytest.approx(0.5)
    assert result["low"] <= 0.5 <= result["high"] and result["low"] > 0.3
    with pytest.raises(ValueError):
        paired_bootstrap([1], [1, 0])


def write_run(root, experiment, benchmark, arm, correct, model="mimo", tokens=100):
    folder = root / experiment / model / benchmark / arm
    folder.mkdir(parents=True)
    lines = [json.dumps({"id": f"q{i}", "correct": c, "status": "ok",
                         "usage": {"calls": 2, "prompt_tokens": tokens, "completion_tokens": 0}})
             for i, c in enumerate(correct)]
    (folder / "predictions.jsonl").write_text("\n".join(lines) + "\n")


def test_aggregate_pairs_extensions_with_the_control_and_skips_pilots(tmp_path):
    write_run(tmp_path, "exp1_baseline", "paraphrased_tomi", "thoughttracing", [1, 0, 0, 1])
    write_run(tmp_path, "exp3_persona", "paraphrased_tomi", "observer", [1, 1, 1, 1])
    write_run(tmp_path, "exp3_persona", "paraphrased_tomi", "observer", [0], model="mimo/pilot-1")
    table, compared = aggregate.main(tmp_path)
    assert len(table) == 2 and len(compared) == 1
    row = compared[0]
    assert (row["n"], row["control_accuracy"], row["variant_accuracy"]) == (4, 0.5, 1.0)
    assert (row["only_variant"], row["only_control"]) == (2, 0)
    out = tmp_path / "aggregate"
    assert (out / "extensions.png").stat().st_size > 0
    assert list(csv.DictReader((out / "main_table.csv").open()))[0]["calls_per_question"] == "2.0"


def test_comparisons_mark_autotom_lead_and_token_cost(tmp_path):
    write_run(tmp_path, "exp1_baseline", "mmtom", "thoughttracing", [1, 0, 0, 0], tokens=100)
    write_run(tmp_path, "exp1_baseline", "mmtom", "autotom", [1, 1, 1, 0], tokens=900)
    write_run(tmp_path, "exp6_summary", "mmtom", "summary", [1, 1, 0, 0], tokens=150)
    _, compared = aggregate.main(tmp_path)
    row = next(r for r in compared if r["experiment"] == "exp6_summary")
    assert row["autotom_lead"] == pytest.approx(0.5) and row["autotom_lead_n"] == 4
    assert (row["control_tokens_per_question"], row["variant_tokens_per_question"]) == (100, 150)
    assert (tmp_path / "aggregate" / "extensions.png").stat().st_size > 0
