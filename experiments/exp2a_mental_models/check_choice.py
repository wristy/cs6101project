"""Check which mental model Jev picks across MMToM-QA stories, before a full Exp 2a run.

    python -m experiments.exp2a_mental_models.check_choice --stories 20

One Jev call per distinct story (the same story text ThoughtTracing sees, without the question).
Prints how often each model wins and writes results/exp2a_choice_check.json.
"""

import argparse
from collections import Counter
import json

from dotenv import load_dotenv

from experiments.exp2a_mental_models.variant import choose_model
from shared.runner.experiment import ROOT, load_benchmark


def stories(limit):
    seen = []
    for example in load_benchmark("mmtom"):
        story = example.native["thoughttracing"]["input"].split("\nQuestion:")[0].strip()
        if story not in seen:
            seen.append(story)
        if len(seen) == limit:
            break
    return seen


def main():
    from shared.models.typesafe import JevClient
    load_dotenv(ROOT / ".env")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--stories", type=int, default=20)
    args = parser.parse_args()
    jev = JevClient(max_calls=args.stories + 5)
    records = [choose_model(jev, story)[1] for story in stories(args.stories)]
    counts = Counter(record["chosen"] for record in records)
    out = ROOT / "results/exp2a_choice_check.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"counts": counts, "records": records, "usage": jev.usage()}, indent=1) + "\n")
    print(f"{len(records)} stories: " + ", ".join(f"{name} {n}" for name, n in counts.most_common()))
    print(f"Jev input tokens: {jev.usage()['jev_input_tokens']}; details in {out}")


if __name__ == "__main__":
    main()
