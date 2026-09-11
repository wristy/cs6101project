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
- AutoToM routes generation and raw `P(A)` likelihood calls to the shared client.
  No GPT-name gate, A/B renormalization, or missing-token fallback is used there.
- Unused AutoToM video and GPU imports are lazy. An unused import of the absent
  `get_logits` function is removed so the released solver can load.

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
source revision or a conflicting patch instead of resetting a checkout.