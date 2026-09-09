"""Run each upstream method in a separate Python process."""

import argparse
import importlib
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np

from tom_research.bridge import configure
from tom_research.client import LLMClient, ModelConfig, ProtocolError


ROOT = Path(__file__).resolve().parents[2]


def load_upstream(method):
    os.environ["TOM_RESEARCH_BACKEND"] = "1"
    if method == "thoughttracing":
        source = ROOT / "external/thought-tracing"
        module = "tracer"
    elif method == "autotom":
        source = ROOT / "external/AutoToM/model"
        module = "ProbSolver"
    else:
        raise ValueError(f"Unknown upstream method: {method}")
    sys.path.insert(0, str(source))
    return importlib.import_module(module)


ANSWER_PROMPT = (
    "The short one-sentence answer specifying the most detailed location including both "
    "the container and the place (e.g., from A in B) without any explanation is:"
)


def run_method(method, payload, client, workdir, upstream=None):
    configure(client)
    workdir = Path(workdir).resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    context, question = payload["context"], payload["question"]
    if method in ("plain", "thoughttracing"):
        trace = ""
        if method == "thoughttracing":
            upstream = upstream or load_upstream(method)
            args = SimpleNamespace(
                model=client.config.model, tracing_model=client.config.model,
                output_dir=str(workdir), run_id="pilot", n_hypotheses=4,
                dataset="tomi", use_helper_llm=False, input_is_chat=False,
                use_perception_only=False, target_perceptions="sight",
                likelihood_estimate="prompting", print=False,
            )
            tracer = upstream.Tracer(args)
            trace = tracer.trace(context, target_agent=payload["target"])
            if not trace.strip():
                raise ProtocolError("ThoughtTracing returned an empty trace")
        prompt = f"{context}\n\nQuestion: {question}\n\n{trace}\n\n{ANSWER_PROMPT}"
        if payload.get("answer_format") == "multiple_choice":
            prompt = (f"{context}\n\nQuestion: {question}\n\n{trace}\n\n"
                      f"Choices: {json.dumps(payload['choices'])}\n"
                      "Return exactly one choice as written, without explanation.")
        answer = client.generate(prompt, max_tokens=256)
        return {"status": "ok", "answer": answer, "trace": trace, "usage": client.usage()}
    if method != "autotom":
        raise ValueError(f"Unknown method {method}")
    upstream = upstream or load_upstream(method)
    candidates = list(payload["choices"])
    if len(candidates) < 2 or len(candidates) != len(set(candidates)):
        raise ValueError("AutoToM needs at least two distinct story-derived location hypotheses")
    solver = upstream.ProblemSolver(
        story=context, question=question, choices=candidates, K=1,
        assigned_model=[], model_name="automated", episode_name="pilot",
        llm=client.config.model, hypo_llm=client.config.model, verbose=False,
        dataset_name="customize", hypo_method="guided", nested=False,
        video_id=None, answerfunc=upstream.argmax, back_inference=True,
        reduce_hypotheses=True, no_model_adjustment=False, recursion_depth=0,
        seed=client.config.seed,
    )
    probabilities, model_record = solver.solve()
    probabilities = np.asarray(probabilities, dtype=float)
    if (probabilities.shape != (len(candidates),) or not np.isfinite(probabilities).all()
            or (probabilities < 0).any() or not np.isclose(probabilities.sum(), 1)):
        raise ProtocolError("AutoToM returned an invalid posterior")
    return {"status": "ok", "answer": candidates[int(probabilities.argmax())],
            "posterior": probabilities.tolist(), "hypotheses": candidates,
            "model_record": model_record, "usage": client.usage()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-import", choices=["thoughttracing", "autotom"])
    parser.add_argument("--request", type=Path)
    parser.add_argument("--result", type=Path)
    args = parser.parse_args()
    if args.check_import:
        load_upstream(args.check_import)
        print(json.dumps({"status": "ok", "method": args.check_import}))
        return
    if args.request is None or args.result is None:
        parser.error("Provide --request and --result, or --check-import")
    request = json.loads(args.request.read_text())
    workdir = args.result.resolve().parent
    runtime = workdir / "model"
    runtime.mkdir(parents=True, exist_ok=True)
    os.chdir(runtime)
    np.random.seed(request["model_config"]["seed"])
    import random
    random.seed(request["model_config"]["seed"])
    client = LLMClient(ModelConfig(**request["model_config"]), log_path=workdir / "calls.jsonl")
    try:
        result = run_method(request["method"], request["payload"], client, workdir)
    except Exception as error:
        result = {"status": "error", "error": f"{type(error).__name__}: {error}", "usage": client.usage()}
    args.result.write_text(json.dumps(result, default=lambda value: value.tolist() if hasattr(value, "tolist") else str(value)))


if __name__ == "__main__":
    main()