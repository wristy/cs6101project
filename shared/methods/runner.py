"""Run one method on one question in an isolated worker process."""

from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import sys

from shared.models.client import ModelConfig


class LiveRunner:
    def __init__(self, config: ModelConfig, variant=None, item_seconds=1200):
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
                return {"status": "error", "error": "Item wall-time budget exceeded; inspect calls.jsonl"}
        if process.returncode or not result_path.exists():
            raise RuntimeError(f"Worker failed; inspect {directory / 'worker.log'}")
        return json.loads(result_path.read_text())