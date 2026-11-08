"""The pilot covers every planned arm and projects cost from per-question usage."""

from shared.runner import pilot


def test_every_planned_arm_is_in_the_pilot():
    runs = pilot.planned_runs()
    names = {(e, a["name"], m, b) for e, a, m, b in runs}
    assert ("exp1_baseline", "thoughttracing", "gpt6sol", "bigtom") in names
    assert ("exp4_scoring", "jev_noul", "mimo", "paraphrased_tomi") in names
    assert len(runs) == 15 + 6 + 1  # Exp 1 (4 MiMo benchmarks + GPT-6 Sol BigToM, 3 arms) + 7 extension arms
    assert {e for e, *_ in pilot.planned_runs(models=["gpt6sol"])} == {"exp1_baseline"}


def test_projection_scales_means_to_the_full_run():
    records = [{"status": "ok", "correct": True, "seconds": 60,
                "usage": {"calls": 10, "prompt_tokens": 1_000_000, "completion_tokens": 0, "embed_tokens": 0}},
               {"status": "error", "correct": False, "seconds": 60,
                "usage": {"calls": 10, "prompt_tokens": 1_000_000, "completion_tokens": 1_000_000,
                          "jev_input_tokens": 100}}]
    row = pilot.project(records, "mimo", full_size=200, workers=4)
    # mean per question: 1M input ($0.39) + 0.5M output ($0.39) = $0.78
    assert row["cost_per_q"] == 0.78 and row["full_cost_usd"] == 156.0
    assert row["errors"] == 1 and row["correct"] == 1 and row["full_jev_input_tokens"] == 10000
    assert row["full_hours"] == round(60 * 200 / 4 / 3600, 2)
