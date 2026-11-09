"""Run one arm (a method, model and optional ThoughtTracing variant) over frozen questions.

Results are append-only and resumable: each finished question is one line in
<directory>/predictions.jsonl, written before the next starts, and a rerun skips finished
questions. manifest.json records the arm and the exact question list; a rerun with a different
one is refused, so one directory always holds one arm.
"""

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
import json
import os
from pathlib import Path
import threading
import time

from shared.runner.scoring import score

METHODS = {"plain", "thoughttracing", "autotom"}


def _read_records(path):
    if not path.exists():
        return []
    try:
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    except json.JSONDecodeError as error:
        raise ValueError(f"Corrupt results; inspect {path} before resuming") from error


def run_arm(examples, arm, runner, directory, session_seconds=14400, workers=1):
    """arm = {"name", "method", "model", "variant" (optional), ...}; runner(method, payload, dir).

    workers > 1 answers that many questions at once (each in its own worker process). Every
    question is still independent, so this changes wall time only; records are written in the
    order questions finish, and summaries do not depend on that order.
    """
    if arm["method"] not in METHODS:
        raise ValueError(f"Unknown method {arm['method']}")
    if arm.get("variant") and arm["method"] != "thoughttracing":
        raise ValueError("Variants change ThoughtTracing only")
    if len({item.id for item in examples}) != len(examples):
        raise ValueError("Duplicate question IDs")
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    manifest = {"arm": arm, "ids": [item.id for item in examples]}
    manifest_path = directory / "manifest.json"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text()) != json.loads(json.dumps(manifest)):
            raise ValueError("Run manifest differs; use a new result directory")
    else:
        manifest_path.write_text(json.dumps(manifest, indent=1) + "\n")
    results_path = directory / "predictions.jsonl"
    records = _read_records(results_path)
    done = {record["id"] for record in records}
    if len(done) != len(records) or not done <= set(manifest["ids"]):
        raise ValueError(f"Duplicate or unexpected results; inspect {results_path}")
    started = time.monotonic()
    lock = threading.Lock()

    def answer(item):
        # Checked when each question starts, so a session stops taking new questions on time.
        if time.monotonic() - started >= session_seconds:
            return
        item_dir = directory / "items" / sha256(item.id.encode()).hexdigest()[:16]
        item_dir.mkdir(parents=True, exist_ok=True)
        item_started = time.monotonic()
        try:
            result = runner(arm["method"], item.inference_input(), item_dir)
        except Exception as error:
            result = {"status": "error", "error": f"{type(error).__name__}: {error}"}
        graded = (score(item, arm["method"], result.get("answer")) if result.get("status") == "ok"
                  else {"status": result.get("status", "error"), "correct": False, "choice": None})
        record = {**result, **graded, "arm": arm["name"], "method": arm["method"], "id": item.id,
                  "benchmark": item.benchmark, "category": item.category,
                  "belief_type": item.belief_type, "gold": item.answer,
                  "seconds": time.monotonic() - item_started}
        with lock:
            with results_path.open("a") as stream:
                stream.write(json.dumps(record, default=str) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            records.append(record)
            done.add(item.id)

    pending = [item for item in examples if item.id not in done]

    if workers <= 1:
        for item in pending:
            answer(item)
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for _ in pool.map(answer, pending):
                pass
    return records


def summarize(records):
    """One row per arm and benchmark: accuracy, its splits, failures and cost per question."""
    groups = defaultdict(list)
    for record in records:
        groups[(record["arm"], record["benchmark"])].append(record)
    rows = []
    for (arm, benchmark), selected in sorted(groups.items()):
        total = len(selected)
        row = {"arm": arm, "benchmark": benchmark, "method": selected[0]["method"], "total": total,
               "correct": sum(r["correct"] for r in selected),
               "errors": sum(r["status"] == "error" for r in selected),
               "invalid": sum(r["status"] == "invalid" for r in selected),
               "ties": sum(bool(r.get("tie")) for r in selected),
               "seconds": sum(r["seconds"] for r in selected)}
        row["accuracy"] = row["correct"] / total
        for key in ("calls", "prompt_tokens", "completion_tokens", "usage_missing"):
            row[key] = sum(r.get("usage", {}).get(key, 0) for r in selected)
        for field in ("belief_type", "category"):
            for value in sorted({r[field] for r in selected if r[field]}):
                subset = [r for r in selected if r[field] == value]
                row[f"accuracy[{value}]"] = sum(r["correct"] for r in subset) / len(subset)
        rows.append(row)
    return rows
