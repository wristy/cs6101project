"""Format check for Experiments 2a/2b: does the model fill the structure, with and without
response_format, and does the provider enforce the schema?

For the first N frozen stories, ask once for 4 hypotheses in the structure without
response_format and once with it, and validate every reply. 2N calls per structure.

    python -m shared.tracing.format_check --experiment 2b --stories 10
"""

import argparse
import json

from dotenv import load_dotenv

from shared.models.client import LLMClient
from shared.models.providers import model_config
from shared.runner.experiment import ROOT, load_benchmark
from shared.tracing.structured import StructureError, list_schema, parse_json, validate


def formats(experiment):
    if experiment == "2b":
        from experiments.exp2b_triples.variant import Triples
        return "paraphrased_tomi", [Triples]
    from experiments.exp2a_mental_models.variant import MODELS
    return "mmtom", MODELS


def check(client, examples, fmt, n=4):
    rows = []
    schema = list_schema(fmt.schema, n)
    for example in examples:
        prompt = (f"{example.context}\n\nGenerate {n} hypotheses on what {example.target} will be thinking "
                  f"(e.g., beliefs). {fmt.instructions(example.target)}\n"
                  f"Answer with JSON only: {{\"hypotheses\": [{n} hypotheses]}}.")
        for enforced in (False, True):
            response_format = ({"type": "json_schema", "json_schema": {"name": "hypotheses", "schema": schema,
                                                                       "strict": True}} if enforced else None)
            row = {"id": example.id, "format": fmt.name, "response_format": enforced}
            try:
                reply = client.generate(prompt, max_tokens=1024, response_format=response_format)
                value = validate(parse_json(reply), schema)
                row.update(valid=True, rendered=[fmt.render(example.target, h) for h in value["hypotheses"]])
            except StructureError as error:
                row.update(valid=False, error=str(error), reply=reply[:500])
            except Exception as error:
                row.update(valid=False, error=f"{type(error).__name__}: {error}")
            rows.append(row)
    return rows


def main():
    load_dotenv(ROOT / ".env")
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", choices=["2a", "2b"], required=True)
    parser.add_argument("--stories", type=int, default=10)
    parser.add_argument("--model", default="mimo")
    args = parser.parse_args()
    benchmark, structures = formats(args.experiment)
    out = ROOT / "results" / f"format_check_{args.experiment}"
    client = LLMClient(model_config(args.model), log_path=out / "calls.jsonl")
    examples = load_benchmark(benchmark)[:args.stories]
    rows = [row for fmt in structures for row in check(client, examples, fmt)]
    (out / "rows.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    for fmt in structures:
        for enforced in (False, True):
            subset = [r for r in rows if r["format"] == fmt.name and r["response_format"] == enforced]
            print(f"{fmt.name:14s} response_format={enforced!s:5s} valid {sum(r['valid'] for r in subset)}/{len(subset)}")


if __name__ == "__main__":
    main()
