"""What every experiments/<exp>/run.py shares: load questions, run one arm, write its summary.

Results go to results/<experiment>/<model>/<benchmark>/<arm>/ (or .../pilot-<n>/... with a
question limit, so a pilot never mixes with a full run). predictions.jsonl and summary.json are
committed; call logs and per-question scratch files are git-ignored.
"""

import argparse
import importlib
import json
from pathlib import Path

from shared.methods.runner import LiveRunner
from shared.models.providers import CHAT_MODELS, model_config
from shared.runner.evaluation import run_arm, summarize

ROOT = Path(__file__).resolve().parents[2]
BENCHMARKS = ("bigtom", "mmtom", "paraphrased_tomi", "exploretom")


def load_benchmark(name):
    if name not in BENCHMARKS:
        raise ValueError(f"Unknown benchmark {name}")
    return importlib.import_module(f"shared.benchmarks.{name}").load()


def result_dir(experiment, model, benchmark, arm, limit=None, root=ROOT / "results"):
    base = Path(root) / experiment / model
    if limit:
        base = base / f"pilot-{limit}"
    return base / benchmark / arm


def run(experiment, arm, model, benchmark, limit=None, runner=None, root=ROOT / "results", **overrides):
    """Run one arm on one benchmark and return its summary row."""
    examples = load_benchmark(benchmark)[:limit] if limit else load_benchmark(benchmark)
    config = model_config(model, **overrides)
    runner = runner or LiveRunner(config, variant=arm.get("variant"))
    directory = result_dir(experiment, model, benchmark, arm["name"], limit, root)
    records = run_arm(examples, {**arm, "model": model, "model_config": config.model}, runner, directory)
    summary = summarize(records)
    (directory / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    return summary


def main(experiment, arms, plan):
    """Command line shared by the run.py files. plan maps model -> benchmarks it runs on."""
    parser = argparse.ArgumentParser(description=f"Run {experiment}")
    parser.add_argument("--model", choices=sorted(CHAT_MODELS), default="mimo")
    parser.add_argument("--benchmark", choices=BENCHMARKS, action="append")
    parser.add_argument("--arm", choices=[arm["name"] for arm in arms], action="append")
    parser.add_argument("--limit", type=int, help="run only the first N frozen questions (a pilot)")
    parser.add_argument("--max-calls", type=int, default=500, help="model-call budget per question")
    args = parser.parse_args()
    benchmarks = args.benchmark or plan[args.model]
    if not set(benchmarks) <= set(plan.get(args.model, ())):
        parser.error(f"{args.model} is planned only for {plan.get(args.model)}")
    for benchmark in benchmarks:
        for arm in arms:
            if args.arm and arm["name"] not in args.arm:
                continue
            for row in run(experiment, arm, args.model, benchmark, args.limit, max_calls=args.max_calls):
                print(json.dumps(row))
