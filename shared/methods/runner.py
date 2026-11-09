"""Run one method on one question in an isolated worker process."""

from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import sys

from shared.models.client import ModelConfig


def usage_from_log(path):
    """Token usage recovered from a worker's calls.jsonl, for a worker that was stopped before it
    could report its own (the call log is written as each call returns)."""
    import ast
    usage = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "seconds": 0.0, "usage_missing": 0,
             "recovered_from_log": True}
    if not Path(path).exists():
        return usage
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get("status") != "ok":
            continue
        usage["calls"] += 1
        usage["seconds"] += float(record.get("seconds") or 0)
        response = record.get("response")
        if isinstance(response, str):
            try:
                response = ast.literal_eval(response)
            except (ValueError, SyntaxError):
                response = None
        reported = (response or {}).get("usage") if isinstance(response, dict) else None
        if not reported:
            usage["usage_missing"] += 1
            continue
        usage["prompt_tokens"] += reported.get("prompt_tokens") or 0
        usage["completion_tokens"] += reported.get("completion_tokens") or 0
    return usage


class LiveRunner:
    # AutoToM's nested reasoning on long ExploreToM stories needs 300-450 calls (~20-25 min with
    # MiMo); the per-question call budget (max_calls) still bounds cost.
    def __init__(self, config: ModelConfig, variant=None, item_seconds=3600):
        self.config = config
        self.variant = variant
        self.item_seconds = item_seconds

    def __call__(self, method, payload, directory):
        directory = Path(directory).resolve()
        request_path = directory / "request.json"
        result_path = directory / "result.json"
        request_path.write_text(json.dumps({"method": method, "payload": payload,
                                            "model_config": asdict(self.config),
                                            "variant": self.variant}))
        with (directory / "worker.log").open("w") as log:
            try:
                process = subprocess.run(
                    [sys.executable, "-m", "shared.methods.worker", "--request", str(request_path),
                     "--result", str(result_path)], stdout=log, stderr=subprocess.STDOUT,
                    timeout=self.item_seconds,
                )
            except subprocess.TimeoutExpired:
                return {"status": "error", "error": "Item wall-time budget exceeded; inspect calls.jsonl",
                        "usage": usage_from_log(directory / "calls.jsonl")}
        if process.returncode or not result_path.exists():
            raise RuntimeError(f"Worker failed; inspect {directory / 'worker.log'}")
        return json.loads(result_path.read_text())