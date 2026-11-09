"""Calibrate Experiment 5's duplicate threshold on hand-labelled hypothesis pairs.

Each Exp 5 trace logs, at every check, the four hypotheses it embedded (calls-embed.jsonl) and
their cosine similarities (rejuvenation_log in the tracer output). No new API calls are needed.

1. Write a labelling sheet from finished Exp 5 runs (pilot or full):
       python -m experiments.exp5_rejuvenation.calibrate sheet results/exp5_rejuvenation/mimo/pilot-10
   It samples pairs spread over the similarity range (more of them near the current threshold)
   and writes results/exp5_calibration/pairs.csv. The sheet does not show the similarity, so
   labels are not anchored on it.
2. Fill the `same` column: y if the two hypotheses say essentially the same thing about the
   agent's mind (one adds nothing the other lacks), n otherwise.
3. Fit:
       python -m experiments.exp5_rejuvenation.calibrate fit
   It prints, for each candidate threshold, how well "cosine > threshold" matches the labels, and
   recommends the one with the best accuracy (ties: fewer false duplicates, then the higher value).
   Run Exp 5 with EXP5_THRESHOLD=<value>.
"""

import argparse
import csv
import json
from pathlib import Path
import random

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/exp5_calibration"
YES, NO = {"y", "yes", "1", "true", "same"}, {"n", "no", "0", "false", "different"}


def _rejuvenation_logs(item_dir):
    for path in sorted(item_dir.glob("tracer-*.jsonl")):
        for line in path.read_text().splitlines():
            if line.strip():
                found = _find(json.loads(line), "rejuvenation_log")
                if found is not None:
                    return found
    return []


def _find(value, key):
    if isinstance(value, dict):
        if key in value:
            return value[key]
        value = list(value.values())
    if isinstance(value, list):
        for child in value:
            found = _find(child, key)
            if found is not None:
                return found
    return None


def collect_pairs(run_dirs):
    """Every within-set pair of embedded hypotheses, with its logged cosine."""
    pairs = []
    for run_dir in run_dirs:
        for item_dir in sorted(Path(run_dir).rglob("items/*")):
            embed_log = item_dir / "calls-embed.jsonl"
            if not embed_log.exists():
                continue
            calls = [json.loads(line) for line in embed_log.read_text().splitlines() if line.strip()]
            logs = _rejuvenation_logs(item_dir)
            if len(calls) != len(logs):
                raise ValueError(f"{item_dir}: {len(calls)} embedding calls but {len(logs)} logged checks")
            for step, (call, entry) in enumerate(zip(calls, logs)):
                texts, similarities = call["request"]["input"], entry["similarities"]
                for j in range(len(texts)):
                    for i in range(j):
                        if texts[i].strip() != texts[j].strip():
                            pairs.append({"source": f"{item_dir}#{step}", "text_a": texts[i], "text_b": texts[j],
                                          "cosine": similarities[i][j]})
    return pairs


def sample_pairs(pairs, n, seed=0, centre=0.85, width=0.06):
    """Half the pairs near the current threshold (centre ± width), the rest spread by cosine."""
    rng = random.Random(seed)
    near = [p for p in pairs if abs(p["cosine"] - centre) <= width]
    far = sorted((p for p in pairs if abs(p["cosine"] - centre) > width), key=lambda p: p["cosine"])
    chosen = rng.sample(near, min(len(near), n // 2))
    rest = n - len(chosen)
    if far and rest > 0:
        step = len(far) / min(rest, len(far))
        chosen += [far[int(k * step)] for k in range(min(rest, len(far)))]
    if len(chosen) < n:
        remaining = [p for p in pairs if p not in chosen]
        chosen += rng.sample(remaining, min(len(remaining), n - len(chosen)))
    rng.shuffle(chosen)
    return chosen


def write_sheet(chosen, out=OUT):
    out.mkdir(parents=True, exist_ok=True)
    with (out / "pairs.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["pair", "hypothesis_a", "hypothesis_b", "same"])
        for number, pair in enumerate(chosen, 1):
            writer.writerow([number, pair["text_a"], pair["text_b"], ""])
    key = {number: {"cosine": pair["cosine"], "source": pair["source"]} for number, pair in enumerate(chosen, 1)}
    (out / "pairs_key.json").write_text(json.dumps(key, indent=1) + "\n")


def read_labels(out=OUT):
    key = json.loads((out / "pairs_key.json").read_text())
    labelled = []
    with (out / "pairs.csv").open(newline="") as stream:
        for row in csv.DictReader(stream):
            mark = row["same"].strip().lower()
            if mark in YES | NO:
                labelled.append((key[row["pair"]]["cosine"], mark in YES))
            elif mark:
                raise ValueError(f"Pair {row['pair']}: label {row['same']!r} is not y or n")
    return labelled


def fit(labelled):
    """Rows for each candidate threshold, and the recommended one."""
    cosines = sorted({c for c, _ in labelled})
    candidates = sorted({round(c, 4) for c in cosines} | {round(c - 1e-4, 4) for c in cosines})
    rows = []
    for threshold in candidates:
        tp = sum(c > threshold and same for c, same in labelled)
        fp = sum(c > threshold and not same for c, same in labelled)
        fn = sum(c <= threshold and same for c, same in labelled)
        tn = len(labelled) - tp - fp - fn
        rows.append({"threshold": threshold, "accuracy": (tp + tn) / len(labelled),
                     "false_duplicates": fp, "missed_duplicates": fn, "flagged": tp + fp})
    best = max(rows, key=lambda r: (r["accuracy"], -r["false_duplicates"], r["threshold"]))
    return rows, best


def main():
    parser = argparse.ArgumentParser(description="Calibrate the Exp 5 duplicate threshold")
    commands = parser.add_subparsers(dest="command", required=True)
    sheet = commands.add_parser("sheet")
    sheet.add_argument("runs", nargs="+", type=Path, help="Exp 5 result folders to read")
    sheet.add_argument("--pairs", type=int, default=30)
    sheet.add_argument("--seed", type=int, default=0)
    commands.add_parser("fit")
    args = parser.parse_args()
    if args.command == "sheet":
        pairs = collect_pairs(args.runs)
        chosen = sample_pairs(pairs, args.pairs, args.seed)
        write_sheet(chosen)
        print(f"{len(pairs)} pairs found; wrote {len(chosen)} to {OUT / 'pairs.csv'}")
        return
    labelled = read_labels()
    if not labelled:
        parser.error(f"No labels in {OUT / 'pairs.csv'}; fill the `same` column with y or n")
    rows, best = fit(labelled)
    print(f"{len(labelled)} labelled pairs, {sum(s for _, s in labelled)} marked the same")
    for row in rows:
        mark = "  <- recommended" if row is best else ""
        print(f"cosine > {row['threshold']:.4f}: accuracy {row['accuracy']:.2f}, "
              f"false duplicates {row['false_duplicates']}, missed {row['missed_duplicates']}{mark}")
    print(f"\nRun Exp 5 with EXP5_THRESHOLD={best['threshold']}")


if __name__ == "__main__":
    main()
