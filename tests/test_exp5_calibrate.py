"""Exp 5 threshold calibration: pairs come from logged traces; the fit matches labels."""

import csv
import json

from experiments.exp5_rejuvenation import calibrate


def make_item(root, texts, similarities):
    item = root / "items" / "abc"
    item.mkdir(parents=True)
    (item / "calls-embed.jsonl").write_text(json.dumps({"request": {"input": texts}}) + "\n")
    trace = {"summary": "", "trace": {"rejuvenation_log": [{"similarities": similarities}]}}
    (item / "tracer-x.jsonl").write_text(json.dumps(trace) + "\n")


def test_sheet_hides_cosine_and_fit_recovers_threshold(tmp_path):
    texts = ["A wants the key", "A wants the key badly", "A thinks the key is lost"]
    sims = [[1, 0.93, 0.7], [0.93, 1, 0.72], [0.7, 0.72, 1]]
    make_item(tmp_path / "run", texts, sims)
    pairs = calibrate.collect_pairs([tmp_path / "run"])
    assert sorted(p["cosine"] for p in pairs) == [0.7, 0.72, 0.93]
    chosen = calibrate.sample_pairs(pairs, 3)
    out = tmp_path / "out"
    calibrate.write_sheet(chosen, out)
    rows = list(csv.DictReader((out / "pairs.csv").open()))
    assert "cosine" not in rows[0] and len(rows) == 3
    key = json.loads((out / "pairs_key.json").read_text())
    for row in rows:
        row["same"] = "y" if key[row["pair"]]["cosine"] > 0.9 else "n"
    with (out / "pairs.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    _, best = calibrate.fit(calibrate.read_labels(out))
    assert best["accuracy"] == 1.0 and 0.72 <= best["threshold"] < 0.93


def test_mismatched_logs_are_refused(tmp_path):
    make_item(tmp_path / "run", ["a", "b"], [[1, 0.5], [0.5, 1]])
    (tmp_path / "run/items/abc/calls-embed.jsonl").write_text("")
    try:
        calibrate.collect_pairs([tmp_path / "run"])
    except ValueError as error:
        assert "embedding calls" in str(error)
    else:
        raise AssertionError("expected a mismatch error")
