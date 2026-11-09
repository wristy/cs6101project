# ThoughtTracing × AutoToM experiments

How much of AutoToM's lead over ThoughtTracing can small, targeted changes to ThoughtTracing recover?
Each experiment changes one ThoughtTracing step. ThoughtTracing stays question-agnostic throughout.

## Layout

| Path | What it holds |
| --- | --- |
| `shared/models/` | One logged client for every model (MiMo on DeepInfra, GPT-6 Sol on OpenAI, Jev, Noul, embeddings) and the A/B probability rule |
| `shared/benchmarks/` | Benchmark loaders; frozen question-ID lists in `shared/benchmarks/data/` |
| `shared/methods/` | Baselines (plain, ThoughtTracing, AutoToM) and the worker that runs one method on one question in its own process |
| `shared/tracing/` | Hooks for swapping one ThoughtTracing step |
| `shared/runner/` | Resumable runs, scoring, per-question records |
| `shared/analysis/` | Paired statistics, main table, extension figure |
| `experiments/expN_*/` | One folder per experiment: `run.py`, plus `variant.py` for Experiments 2–6 (see `experiments/README.md`) |
| `results/` | Per-question records per experiment and arm; `results/aggregate/` for tables and figures |
| `external/` + `patches/` | The official ThoughtTracing and AutoToM code at pinned commits, with small patches that route their LLM calls to our client (see `patches/README.md`) |

## Setup

Python 3.12+, Git and `uv`:

```sh
bash scripts/bootstrap.sh      # fetch pinned upstream code, apply patches, install, run offline tests
cp .env.example .env           # then fill in the API keys
```

## Tests

```sh
.venv/bin/python -m pytest -q          # offline tests, no API calls
.venv/bin/python -m pytest -q --live   # also a few real API calls; each skips if its key is missing
```
