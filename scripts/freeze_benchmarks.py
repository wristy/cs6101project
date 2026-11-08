"""Write the frozen question-ID lists in shared/benchmarks/data/.

Run once; the lists are committed. Re-running must give identical files (tests check this).
    python scripts/freeze_benchmarks.py [bigtom mmtom paraphrased_tomi exploretom]
"""

import sys

import pandas as pd

from shared.benchmarks import bigtom, exploretom, mmtom, paraphrased_tomi
from shared.benchmarks.common import freeze


def freeze_bigtom():
    return freeze("bigtom", bigtom.load_all(), bigtom.SOURCE,
                  "All 249 questions of ThoughtTracing's bigtom_agree90.json, in file order.")


def freeze_mmtom():
    examples = mmtom.load_all()
    order = pd.DataFrame(index=range(len(examples))).sample(n=200, random_state=123).index
    return freeze("mmtom", [examples[i] for i in order], mmtom.SOURCE,
                  "ThoughtTracing's sample: DataFrame.sample(n=200, random_state=123) over the 600 rows.")


def freeze_paraphrased_tomi(per_type=25):
    # ThoughtTracing samples over every question in the file (including reality and memory ones).
    total = len(paraphrased_tomi.SOURCE.with_suffix(".trace").read_text().splitlines())
    by_index = {int(example.id.split(":")[1]): example for example in paraphrased_tomi.load_all()}
    remaining = {question_type: per_type for question_type in paraphrased_tomi.TYPES}
    chosen = []
    for index in pd.DataFrame(index=range(total)).sample(n=2000, random_state=123).index:
        example = by_index.get(index)
        if example is not None and remaining[example.category] > 0:
            remaining[example.category] -= 1
            chosen.append(example)
    if any(remaining.values()):
        raise ValueError(f"Not enough questions of each type: {remaining}")
    return freeze("paraphrased_tomi", chosen, paraphrased_tomi.SOURCE,
                  "ThoughtTracing's order: DataFrame.sample(n=2000, random_state=123), first 25 of each "
                  "first/second-order type.")


def freeze_exploretom():
    return exploretom.freeze_selection()


FREEZERS = {"bigtom": freeze_bigtom, "mmtom": freeze_mmtom,
            "paraphrased_tomi": freeze_paraphrased_tomi, "exploretom": freeze_exploretom}


if __name__ == "__main__":
    for name in sys.argv[1:] or ["bigtom", "mmtom", "paraphrased_tomi"]:
        print(FREEZERS[name]())
