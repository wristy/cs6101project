# Experiments

One folder per experiment. Each holds:

- `run.py`: runs the experiment's arms through the shared runner and writes results to `results/<experiment>/<arm>/`.
- `variant.py` (Experiments 2–6): a small subclass of ThoughtTracing's `Tracer` that replaces one step.

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
