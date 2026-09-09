"""Free-text scoring and append-only, resumable pilot results."""

from dataclasses import asdict
from hashlib import sha256
import json
import os
from pathlib import Path
import time


def score_answer(prediction, answer):
    return bool(isinstance(prediction, str) and answer.strip() and answer.lower() in prediction.lower())


def run_pilot(examples, methods, runner, directory, config):
    directory = Path(directory).resolve()
    if len({item.id for item in examples}) != len(examples) or len(set(methods)) != len(methods):
        raise ValueError("Duplicate item IDs or methods")
    if any(method not in {"plain", "thoughttracing", "autotom"} for method in methods):
        raise ValueError("Unknown method")
    track = config.get("track")
    if track is not None:
        formats = {"thoughttracing": "free_text", "autotom": "multiple_choice"}
        if (track not in formats or any(method not in {"plain", track} for method in methods)
                or any(item.answer_format != formats[track] for item in examples)):
            raise ValueError("Methods or dataset format do not match the track")
    manifest = json.loads(json.dumps({"config": config, "methods": methods,
                                     "examples": [asdict(item) for item in examples]}))
    directory.mkdir(parents=True, exist_ok=True)
    manifest_path = directory / "manifest.json"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text()) != manifest:
            raise ValueError("Run manifest differs; use a new result directory")
    else:
        with manifest_path.open("x") as stream:
            json.dump(manifest, stream, indent=2)
    results_path = directory / "predictions.jsonl"
    try:
        results = [json.loads(line) for line in results_path.read_text().splitlines()] if results_path.exists() else []
        completed = {(record["method"], record["id"]) for record in results}
        expected = {(method, item.id) for method in methods for item in examples}
        if len(completed) != len(results) or not completed <= expected:
            raise ValueError("Duplicate or unexpected results")
    except (ValueError, KeyError) as error:
        raise ValueError("Corrupt results; inspect predictions.jsonl before resuming") from error
    started = time.monotonic()
    session_seconds = config.get("session_seconds", 14400)
    for item in examples:
        for method in methods:
            if (method, item.id) in completed:
                continue
            if time.monotonic() - started >= session_seconds:
                return results
            key = sha256(f"{method}:{item.id}".encode()).hexdigest()[:16]
            item_dir = directory / method / key
            item_dir.mkdir(parents=True, exist_ok=True)
            item_started = time.monotonic()
            try:
                result = runner(method, item.inference_input(), item_dir)
                if result.get("status") == "ok" and not str(result.get("answer", "")).strip():
                    result["status"] = "invalid"
            except Exception as error:
                result = {"status": "error", "error": f"{type(error).__name__}: {error}"}
            if item.answer_format == "multiple_choice" and result.get("status") == "ok":
                if result.get("answer", "").strip() not in item.choices:
                    result["status"] = "invalid"
            correct = (result.get("answer", "").strip() == item.answer
                       if item.answer_format == "multiple_choice" and isinstance(result.get("answer"), str)
                       else score_answer(result.get("answer"), item.answer))
            record = {**result, "method": method, "id": item.id, "belief_type": item.belief_type,
                      "track": config.get("track"), "answer_format": item.answer_format,
                      "gold": item.answer, "seconds": time.monotonic() - item_started,
                      "correct": result.get("status") == "ok" and correct}
            with results_path.open("a") as stream:
                stream.write(json.dumps(record) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            results.append(record)
            completed.add((method, item.id))
    return results


def summarize(records):
    reports = []
    groups = {(record.get("track") or "legacy", record["method"]) for record in records}
    for track, method in sorted(groups):
        selected = [record for record in records if record["method"] == method
                    and (record.get("track") or "legacy") == track]
        correct = sum(record["correct"] for record in selected)
        synthetic = sum(record["status"] == "synthetic" for record in selected)
        report = {"track": track, "method": method, "total": len(selected), "correct": correct,
                  "accuracy": None if synthetic else correct / len(selected),
                  "errors": sum(record["status"] == "error" for record in selected),
                  "invalid": sum(record["status"] == "invalid" for record in selected),
                  "synthetic": synthetic, "seconds": sum(record["seconds"] for record in selected)}
        for key in ("calls", "prompt_tokens", "completion_tokens", "usage_missing"):
            report[key] = sum(record.get("usage", {}).get(key, 0) for record in selected)
        for belief in ("true_belief", "false_belief"):
            subset = [record for record in selected if record["belief_type"] == belief]
            report[belief + "_accuracy"] = (sum(record["correct"] for record in subset) / len(subset)
                                            if subset and not synthetic else None)
        reports.append(report)
    return reports