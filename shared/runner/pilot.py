"""Pilot every planned arm on the first N frozen questions and project the full run.

    python -m shared.runner.pilot                 # N=10, every experiment, both models
    python -m shared.runner.pilot --limit 3 --experiment exp1_baseline

One command runs each experiment's PLAN (Exp 1 for MiMo on four benchmarks plus GPT-6 Sol on
BigToM, and each extension on its benchmark) on the same first N questions, so extension arms are
paired with the Exp 1 ThoughtTracing control. Results go to the usual pilot-N folders and resume
like any run. It then prints, per arm: accuracy, errors and invalid answers, mean calls and tokens
per question, and the projected cost and wall time of the full run (all frozen questions) from
those means. It writes the same table to results/pilot-N.csv.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import importlib
import json
import time

from dotenv import load_dotenv

from shared.models.providers import PRICES
from shared.runner import experiment
from shared.runner.experiment import ROOT

EXPERIMENTS = ["exp1_baseline", "exp2a_mental_models", "exp2b_triples", "exp3_persona", "exp4_scoring",
               "exp5_rejuvenation", "exp6_summary"]


def planned_runs(experiments=EXPERIMENTS, models=None):
    """(experiment, arm, model, benchmark) for every arm each experiment plans."""
    runs = []
    for name in experiments:
        module = importlib.import_module(f"experiments.{name}.run")
        for model, benchmarks in module.PLAN.items():
            if models and model not in models:
                continue
            runs += [(module.EXPERIMENT, arm, model, benchmark) for benchmark in benchmarks for arm in module.ARMS]
    return runs


def project(records, model, full_size, workers):
    """Means per question from pilot records, scaled to the full question list."""
    n = len(records)
    usage = [r.get("usage") or {} for r in records]
    mean = lambda key: sum(u.get(key, 0) or 0 for u in usage) / n
    price = PRICES[model]
    chat = mean("prompt_tokens") * price["input"] / 1e6 + mean("completion_tokens") * price["output"] / 1e6
    embed = mean("embed_tokens") * PRICES["embedding"]["input"] / 1e6
    seconds = sum(r["seconds"] for r in records) / n
    return {"n": n, "correct": sum(r["correct"] for r in records),
            "errors": sum(r["status"] == "error" for r in records),
            "invalid": sum(r["status"] == "invalid" for r in records),
            "ties": sum(bool(r.get("tie")) for r in records),
            "calls_per_q": round(mean("calls"), 1), "prompt_tokens_per_q": round(mean("prompt_tokens")),
            "completion_tokens_per_q": round(mean("completion_tokens")),
            "score_fallbacks": sum(u.get("score_fallbacks", 0) or 0 for u in usage),
            "truncated": sum(u.get("truncated", 0) or 0 for u in usage),
            "seconds_per_q": round(seconds, 1), "cost_per_q": round(chat + embed, 5),
            "full_questions": full_size, "full_cost_usd": round((chat + embed) * full_size, 2),
            "full_jev_input_tokens": round(mean("jev_input_tokens") * full_size),
            "full_hours": round(seconds * full_size / workers / 3600, 2)}


def run_one(spec, limit, workers, root):
    name, arm, model, benchmark = spec
    experiment.run(name, arm, model, benchmark, limit, root=root, workers=workers)
    path = experiment.result_dir(name, model, benchmark, arm["name"], limit, root) / "predictions.jsonl"
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    row = {"experiment": name, "model": model, "benchmark": benchmark, "arm": arm["name"]}
    row.update(project(records, model, len(experiment.load_benchmark(benchmark)), workers))
    first_error = next((r.get("error") for r in records if r["status"] != "ok" and r.get("error")), None)
    row["first_error"] = (first_error or "")[:200]
    return row


def main():
    load_dotenv(ROOT / ".env")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--limit", type=int, default=10, help="questions per arm")
    parser.add_argument("--workers", type=int, default=4, help="questions at once within an arm")
    parser.add_argument("--arms-at-once", type=int, default=4, help="arms run side by side")
    parser.add_argument("--experiment", action="append", choices=EXPERIMENTS)
    parser.add_argument("--model", action="append", choices=sorted(PRICES.keys() - {"embedding"}))
    args = parser.parse_args()
    runs = planned_runs(args.experiment or EXPERIMENTS, args.model)
    root = ROOT / "results"
    print(f"Pilot: {len(runs)} arms x {args.limit} questions", flush=True)
    started = time.monotonic()
    rows = []
    with ThreadPoolExecutor(max_workers=args.arms_at_once) as pool:
        futures = {pool.submit(run_one, spec, args.limit, args.workers, root): spec for spec in runs}
        for future, spec in futures.items():
            try:
                row = future.result()
            except Exception as error:   # a refused manifest or missing key: report, keep going
                row = {"experiment": spec[0], "model": spec[2], "benchmark": spec[3], "arm": spec[1]["name"],
                       "first_error": f"{type(error).__name__}: {error}"[:200]}
            rows.append(row)
            print(json.dumps(row), flush=True)
    out = root / f"pilot-{args.limit}.csv"
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with out.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    total = sum(row.get("full_cost_usd", 0) for row in rows)
    jev = sum(row.get("full_jev_input_tokens", 0) for row in rows)
    problems = [r for r in rows if r.get("errors") or r.get("invalid") or "n" not in r]
    print(f"\nPilot took {(time.monotonic() - started) / 60:.1f} min; table in {out}")
    print(f"Projected full run: ${total:.2f} for the chat and embedding models, plus {jev:,} Jev input tokens")
    print(f"Arms with errors or invalid answers: {len(problems)}"
          + "".join(f"\n  {r['experiment']}/{r['model']}/{r['benchmark']}/{r['arm']}: {r.get('first_error', '')}"
                    for r in problems))


if __name__ == "__main__":
    main()
