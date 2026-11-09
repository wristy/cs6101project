# Experiments

One folder per experiment. Each holds:

- `run.py`: runs the experiment's arms through the shared runner and writes results to `results/<experiment>/<model>/<benchmark>/<arm>/` (pilots under `.../<model>/pilot-N/...`).
- `variant.py` (Experiments 2–6): a small subclass of ThoughtTracing's `Tracer` that replaces one step.

Run one experiment with `python -m experiments.<folder>.run [--limit N] [--workers N]`, or every
planned arm at once with `python -m shared.runner.pilot --limit N` (`--limit 0` for all questions),
which also projects cost and time. Exp 5's duplicate threshold was calibrated with
`experiments/exp5_rejuvenation/calibrate.py`.

| Folder | Experiment | Benchmark | Changes |
| --- | --- | --- | --- |
| `exp1_baseline` | 1. Baseline comparison | all four | nothing: Plain, AutoToM, ThoughtTracing |
| `exp2a_mental_models` | 2a. Mental-model slots | MMToM-QA | how hypotheses are generated and updated |
| `exp2b_triples` | 2b. Relation triples | ParaphrasedToMi | how hypotheses are generated and updated |
| `exp3_persona` | 3. Observer persona | ParaphrasedToMi | the update step's system prompt |
| `exp4_scoring` | 4. Probability scoring | ParaphrasedToMi | how hypotheses are weighted |
| `exp5_rejuvenation` | 5. Embedding rejuvenation | MMToM-QA | how near-duplicates are detected and replaced |
| `exp6_summary` | 6. Trajectory summary | MMToM-QA | the history the update and weighting steps read |

Experiments 2–6 use Experiment 1's ThoughtTracing run as their control, read from `results/exp1_baseline/`.
