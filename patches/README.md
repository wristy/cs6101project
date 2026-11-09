# Upstream Compatibility

The official sources stay in ignored `external/` checkouts. These patches and
their pinned commits reproduce our text-only backend integration. Upstream
licenses remain in the checkouts: MIT for ThoughtTracing, Apache-2.0 for AutoToM.

| Source | Commit |
| --- | --- |
| ThoughtTracing | `966eb2712714aa29b4d5d2395e39af94a2be90d7` |
| AutoToM | `3f569b7ab1d0ee2702ab43ebdc724679d5fc5231` |

Changes are enabled by `TOM_RESEARCH_BACKEND=1` inside isolated method workers:

- ThoughtTracing loads the shared `interact`/`batch_interact` agent without
  importing unused cloud or GPU clients.
- AutoToM routes every LLM call to the shared client: generation, the A/B
  likelihoods (`probs.py`, `utils.py`) and the question classifiers and initial
  model proposal in `model_adjustment.gpt_call`, which upstream sends to GPT-4o.
  When the first token is neither A nor B, the likelihood falls back to 0.1, as
  AutoToM itself does for GPT-4o; each fallback is counted.
- `ProbSolver.py` reads `self.back_inference` instead of the `__main__` global
  `args`, which does not exist when the solver is imported.
- Unused AutoToM video and GPU imports are lazy. An unused import of the absent
  `get_logits` function is removed so the released solver can load.

As a backstop, the worker blocks any remaining direct OpenAI call in AutoToM
modules and removes `OPENAI_API_KEY` from the environment once our own client
holds its key, so an unpatched path fails instead of reaching a real API.

No particle propagation, likelihood prompt, Bayesian marginalization, or model
adjustment algorithm is replaced. The model backend and preprocessing model are
intentional adaptations. ThoughtTracing's released rejuvenation uses a diversity
threshold and requests temperature 1; we preserve that code behavior even though
the paper describes greedy decoding throughout.

After changing an upstream compatibility file, review its diff, run the focused
tests, then regenerate the patches with:

```sh
.venv/bin/python scripts/export_patches.py
```

Do not export unrelated changes. `scripts/bootstrap.sh` refuses an unexpected
source revision. When an older version of a patch is applied, it restores the
pinned files and applies the current patch.