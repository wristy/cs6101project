"""Paired statistics on per-question correctness (1/0), matched by question ID."""

from math import comb

import numpy as np


def paired_bootstrap(control, variant, reps=10000, seed=0):
    """95% percentile interval for mean(variant) - mean(control), resampling questions."""
    control, variant = np.asarray(control, dtype=float), np.asarray(variant, dtype=float)
    if control.shape != variant.shape or control.size == 0:
        raise ValueError("Need two equal-length, non-empty paired samples")
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, control.size, size=(reps, control.size))
    diffs = variant[picks].mean(axis=1) - control[picks].mean(axis=1)
    low, high = np.percentile(diffs, [2.5, 97.5])
    return {"diff": float(variant.mean() - control.mean()), "low": float(low), "high": float(high)}


def mcnemar_exact(control, variant):
    """Two-sided exact McNemar test on the discordant pairs."""
    control, variant = np.asarray(control, dtype=bool), np.asarray(variant, dtype=bool)
    only_control = int((control & ~variant).sum())
    only_variant = int((~control & variant).sum())
    n = only_control + only_variant
    if n == 0:
        return {"only_control": 0, "only_variant": 0, "p": 1.0}
    tail = sum(comb(n, k) for k in range(min(only_control, only_variant) + 1)) / 2 ** n
    return {"only_control": only_control, "only_variant": only_variant, "p": min(1.0, 2 * tail)}
