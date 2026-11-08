"""Run each upstream method in a separate Python process."""

import argparse
import importlib
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np

from shared.bridge import configure
from shared.methods import prompts
from shared.models.client import LLMClient, ModelConfig, ProtocolError


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


def prepare_autotom_runtime(runtime, model):
    """AutoToM reads prompts/prompts_<llm>/ relative to the working directory and ships only the
    GPT-4o set, which it uses for every model. Point prompts/prompts_<our model> at that set."""
    link = Path(runtime) / "prompts" / f"prompts_{model}"
    if not link.exists():
        link.parent.mkdir(parents=True, exist_ok=True)
        link.symlink_to(ROOT / "external/AutoToM/model/prompts/prompts_gpt-4o", target_is_directory=True)
    return link


def load_variant(spec, upstream):
    """A variant is "package.module", whose build(upstream) returns a Tracer subclass."""
    if not spec:
        return upstream.Tracer
    tracer_class = importlib.import_module(spec).build(upstream)
    if not issubclass(tracer_class, upstream.Tracer):
        raise TypeError(f"{spec}.build() must return a subclass of ThoughtTracing's Tracer")
    return tracer_class


def tracer_args(client, workdir, dataset):
    """The defaults of ThoughtTracing's command line (tracer.py), with our model and dataset."""
    return SimpleNamespace(
        model=client.config.model, tracing_model=client.config.model, output_dir=str(workdir),
        run_id="run", n_hypotheses=4, dataset=dataset, use_helper_llm=False, input_is_chat=False,
        use_perception_only=False, target_perceptions="sight", likelihood_estimate="prompting",
        tracer_type="tracer", print=False,
    )


def run_method(method, payload, client, workdir, upstream=None, variant=None):
    """Run one method on one question. The payload is Example.inference_input(): no gold answer."""
    configure(client)
    workdir = Path(workdir).resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    native = payload["native"]
    if method in ("plain", "thoughttracing"):
        dataset, tt_input = native["thoughttracing"]["dataset"], native["thoughttracing"]["input"]
        trace = ""
        if method == "plain":
            prompt = prompts.plain_prompt(dataset, tt_input)
        else:
            upstream = upstream or load_upstream(method)
            tracer = load_variant(variant, upstream)(tracer_args(client, workdir, dataset))
            # As in ThoughtTracing's scripts: the whole input goes in and the tracer finds the
            # target agent itself; on MMToM-QA the script finds it from the question alone.
            if dataset == "mmtom":
                target = tracer.identify_target("Question:" + tt_input.split("\nQuestion:")[-1])
                trace = tracer.trace(tt_input, target_agent=target)
            else:
                trace = tracer.trace(tt_input)
            if not isinstance(trace, str) or not trace.strip():
                raise ProtocolError("ThoughtTracing returned an empty trace")
            prompt = prompts.traced_prompt(dataset, tt_input, trace)
        answer = client.generate(prompt, max_tokens=prompts.MAX_TOKENS[dataset])
        return {"status": "ok", "answer": answer, "trace": trace, "usage": client.usage()}
    if method != "autotom":
        raise ValueError(f"Unknown method {method}")
    upstream = upstream or load_upstream(method)
    settings = native["autotom"]
    candidates = list(payload["choices"])
    if len(candidates) < 2 or len(candidates) != len(set(candidates)):
        raise ValueError("AutoToM needs at least two distinct answer options")
    question = payload["question"]
    # The arguments of AutoToM's own main() (ProbSolver.py) with --automated.
    solver = upstream.ProblemSolver(
        story=payload["context"], question=question, choices=candidates, K=1,
        assigned_model=["State", "Observation", "Belief", "Action", "Goal"],
        model_name="automated", episode_name=payload["id"], llm=client.config.model,
        hypo_llm=client.config.model, verbose=False, dataset_name=settings["dataset_name"],
        hypo_method="guided", nested=None, video_id=None,
        answerfunc=upstream.argmin if "LEAST likely" in question else upstream.argmax,
        back_inference=True, reduce_hypotheses=True,
        precomputed_states=settings.get("states"), precomputed_actions=settings.get("actions"),
        prev_hyp=None, no_model_adjustment=False, recursion_depth=0, use_all_timesteps=False,
        seed=client.config.seed,
    )
    probabilities, model_record = solver.solve()
    if probabilities is None:
        raise ProtocolError("AutoToM could not answer with its chosen model")
    probabilities = np.asarray(probabilities, dtype=float)
    if (probabilities.shape != (len(candidates),) or not np.isfinite(probabilities).all()
            or (probabilities < 0).any() or not np.isclose(probabilities.sum(), 1)):
        raise ProtocolError("AutoToM returned an invalid posterior")
    pick = int(probabilities.argmin() if "LEAST likely" in question else probabilities.argmax())
    return {"status": "ok", "answer": candidates[pick],
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
    if request["method"] == "autotom":
        prepare_autotom_runtime(runtime, request["model_config"]["model"])
    os.chdir(runtime)
    np.random.seed(request["model_config"]["seed"])
    import random
    random.seed(request["model_config"]["seed"])
    client = LLMClient(ModelConfig(**request["model_config"]), log_path=workdir / "calls.jsonl")
    try:
        result = run_method(request["method"], request["payload"], client, workdir,
                            variant=request.get("variant"))
    except Exception as error:
        result = {"status": "error", "error": f"{type(error).__name__}: {error}", "usage": client.usage()}
    args.result.write_text(json.dumps(result, default=lambda value: value.tolist() if hasattr(value, "tolist") else str(value)))


if __name__ == "__main__":
    main()