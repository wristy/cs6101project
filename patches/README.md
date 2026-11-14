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
- Generated replies cut off at `max_tokens` are kept (and counted as truncated),
  as on ThoughtTracing's path: AutoToM parses only the start of each reply.

Upstream bugs found in the full run, each reproduced offline from saved call
logs and fixed with the smallest change that keeps the intended behavior:

- BigToM questions where the agent has not acted yet (`BayesianInference.infer`,
  `init_belief`): the forward-action branch stored the initial-belief prior as a
  scalar that `hypo_propagation` then indexes, and the belief branch returned 3
  values where every caller unpacks 4. The prior is now a one-element array and
  the belief branch returns the same 4-tuple as every other path. As released,
  AutoToM could not answer these question types at all.
- Second-order and nested questions (`Nested.get_nested_states`): one
  `update_state` call omits its required `dataset_name`; the timestep alignment
  indexes the inferred agent's action column, which the extracted table can
  lack; and nested episode names embed a whole belief sentence, which can pass
  the 255-byte file-name limit (now shortened with a hash).
- Reply shapes the parsers did not handle: a bare sentence where
  `hypothesis_generation(_no_observation)` asks for `["..."]` is read as a
  one-item list, and an extraction reply that drops its leading `"A"` flag is
  read as the evidence it contains.

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