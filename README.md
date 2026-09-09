# ToM Research Pilot

Two separate tracks test new models with the released methods:

| Track | Dataset | Answers | Runs |
| --- | --- | --- | --- |
| `thoughttracing` | ThoughtTracing's ParaphrasedToMi | Free text | Plain model, ThoughtTracing |
| `autotom` | AutoToM's released ToMi | Supplied container choices | Plain model, AutoToM |

Each track has 2 debugging questions and 20 fixed pilot questions: 10 true-belief
and 10 false-belief questions, all first-order. Debug and pilot story texts are
disjoint. Choices for AutoToM come from `containers` metadata, not sentence endings.

## Notebooks

Use the existing `.venv` and the `tom-research` notebook kernel.
Run `notebooks/00_data.ipynb`, then `01_debug.ipynb`, then `02_pilot.ipynb`.
Set `TRACK` to `thoughttracing` or `autotom` in each notebook's setup cell.
Alternatively, set `TOM_TRACK` before starting Jupyter. Keep `LIVE = False`.

Offline mode uses synthetic records with no accuracy; it does not execute a model.
Each track writes 40 pilot records under `results/offline/<track>/pilot/`.
Track-specific sample manifests are in `data/<track>_pilot_ids.json`.
The old `data/pilot_ids.json` and any mixed-track results are left untouched and
are not used by the new notebooks.

Prepare both samples and run tests:

```sh
.venv/bin/python -m tom_research.study prepare
.venv/bin/python -m pytest tests -q
```

Development uses red/green TDD: add a failing behavioral test, run it, implement
the smallest change, then rerun the same test. Notebook tests execute both tracks
in fresh kernels. Existing pinned source checkouts in `external/` are required
for dataset and upstream integration tests.

## Interpretation

Compare each new-model result with its own plain baseline. Do not rank methods
across these datasets, or compare a 20-question pilot directly with full published
scores. First reproduce the paper's complete evaluation subset, prompts, scoring,
and decoding configuration. Rerun an original backend where accessible before
attributing a difference solely to the newer model.

Known adaptations remain explicit:

- ThoughtTracing uses Qwen throughout; its paper used GPT-4o for action labeling
  on ParaphrasedToMi. Its free-text scorer accepts a case-insensitive gold substring.
- AutoToM keeps supplied choices and automated Bayesian inference. Its plain
  baseline is prompted to output exactly one container; invalid text is counted
  separately. This formatting protocol is not yet verified as the paper's baseline.
- Qwen thinking is disabled. This is not a test of its default thinking mode.
- The pilots omit higher-order, memory, and reality questions. They cannot
  reproduce either paper's aggregate benchmark score.

Inspect live errors and raw answers, not only accuracy. Saved manifests include
model settings, source hashes and commits; changed settings require a new result
directory. Baselines from different tracks are never pooled in summaries.

## Live Runs

No weight downloads or GPU jobs are part of offline setup. Live runs require an
allocated GPU, a verified endpoint, pinned model revision, and recorded hardware
and server versions in `configs/pilot.json`. Complete the selected track's live
debug notebook first. The pilot refuses mismatched debug settings or datasets.

Official sources: [ThoughtTracing](https://github.com/skywalker023/thought-tracing)
and [AutoToM](https://github.com/SCAI-JHU/AutoToM).