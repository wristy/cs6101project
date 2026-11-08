"""Collect every full run under results/ into the main table, the paired comparisons and the figure.

    python -m shared.analysis.aggregate            # writes results/aggregate/

Only full runs count (pilot-<n> folders are skipped). An error or invalid answer counts as
wrong. Extensions are compared with Experiment 1's MiMo ThoughtTracing run on the same
benchmark, paired on the questions both have.
"""

import csv
import json
from pathlib import Path

from shared.analysis.stats import mcnemar_exact, paired_bootstrap

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
CONTROL = ("exp1_baseline", "mimo", "thoughttracing")
BENCHMARK_NAMES = {"bigtom": "BigToM", "mmtom": "MMToM-QA", "paraphrased_tomi": "ParaphrasedToMi",
                   "exploretom": "ExploreToM"}


def load_runs(root=RESULTS):
    """{(experiment, model, benchmark, arm): {question id: record}} for full runs."""
    runs = {}
    for path in sorted(Path(root).glob("*/*/*/*/predictions.jsonl")):
        arm, benchmark, model, experiment = path.parts[-2], path.parts[-3], path.parts[-4], path.parts[-5]
        if experiment in ("aggregate", "live-tests") or model.startswith("pilot-") or benchmark.startswith("pilot-"):
            continue
        records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        runs[(experiment, model, benchmark, arm)] = {record["id"]: record for record in records}
    return runs


def main_table(runs):
    rows = []
    for (experiment, model, benchmark, arm), records in sorted(runs.items()):
        values = list(records.values())
        calls = [r.get("usage", {}).get("calls", 0) for r in values]
        rows.append({"experiment": experiment, "model": model, "benchmark": benchmark, "arm": arm,
                     "n": len(values), "accuracy": sum(r["correct"] for r in values) / len(values),
                     "errors": sum(r["status"] == "error" for r in values),
                     "invalid": sum(r["status"] == "invalid" for r in values),
                     "calls_per_question": sum(calls) / len(values),
                     "prompt_tokens_per_question": sum(r.get("usage", {}).get("prompt_tokens", 0) for r in values) / len(values),
                     "completion_tokens_per_question": sum(r.get("usage", {}).get("completion_tokens", 0) for r in values) / len(values)})
    return rows


def comparisons(runs, reps=10000):
    rows = []
    for (experiment, model, benchmark, arm), records in sorted(runs.items()):
        if experiment == CONTROL[0]:
            continue
        control = runs.get((CONTROL[0], CONTROL[1], benchmark, CONTROL[2]))
        if control is None:
            continue
        ids = sorted(set(control) & set(records))
        if not ids:
            continue
        a = [bool(control[i]["correct"]) for i in ids]
        b = [bool(records[i]["correct"]) for i in ids]
        interval = paired_bootstrap(a, b, reps=reps)
        test = mcnemar_exact(a, b)
        rows.append({"experiment": experiment, "arm": arm, "benchmark": benchmark, "n": len(ids),
                     "control_accuracy": sum(a) / len(ids), "variant_accuracy": sum(b) / len(ids),
                     **interval, **test})
    return rows


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def figure(rows, path):
    """Dot-and-whisker: each extension's change in accuracy against the control, with 95% CI."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not rows:
        return
    ink, muted, series, surface, grid = "#0b0b0b", "#52514e", "#2a78d6", "#fcfcfb", "#e4e3df"
    labels = [f"{r['experiment'].split('_')[0].replace('exp', 'Exp ')} · {r['arm'].replace('_', ' ')}"
              f"  ({BENCHMARK_NAMES.get(r['benchmark'], r['benchmark'])})" for r in rows]
    y = list(range(len(rows)))[::-1]
    fig, ax = plt.subplots(figsize=(7, 0.45 * len(rows) + 1.2), dpi=200)
    fig.patch.set_facecolor(surface)
    ax.set_facecolor(surface)
    ax.axvline(0, color=muted, linewidth=1)
    for yi, r in zip(y, rows):
        ax.plot([100 * r["low"], 100 * r["high"]], [yi, yi], color=series, linewidth=2, solid_capstyle="round")
        ax.plot(100 * r["diff"], yi, "o", color=series, markersize=7, markeredgecolor=surface, markeredgewidth=2)
        ax.annotate(f"{100 * r['diff']:+.1f}", (100 * r["high"], yi), xytext=(6, 0), textcoords="offset points",
                    va="center", fontsize=8, color=muted)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9, color=ink)
    ax.set_xlabel("Change in accuracy vs ThoughtTracing (points, 95% paired bootstrap CI)", fontsize=9, color=muted)
    ax.tick_params(axis="x", colors=muted, labelsize=8)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color=grid, linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title("Extensions against the ThoughtTracing control (MiMo)", loc="left", fontsize=10, color=ink)
    fig.tight_layout()
    fig.savefig(path, facecolor=surface)
    plt.close(fig)


def main(root=RESULTS, out=None):
    out = Path(out) if out else Path(root) / "aggregate"
    out.mkdir(parents=True, exist_ok=True)
    runs = load_runs(root)
    table, compared = main_table(runs), comparisons(runs)
    write_csv(out / "main_table.csv", table)
    write_csv(out / "comparisons.csv", compared)
    figure(compared, out / "extensions.png")
    return table, compared


if __name__ == "__main__":
    table, compared = main()
    print(f"{len(table)} runs, {len(compared)} comparisons written to results/aggregate/")
