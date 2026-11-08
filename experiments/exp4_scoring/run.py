"""Experiment 4: how ThoughtTracing weights hypotheses, on ParaphrasedToMi.

The original verbal six-level weighting is Experiment 1's MiMo ThoughtTracing run.

    python -m experiments.exp4_scoring.run --arm autotom_ab --limit 2
"""

from shared.runner.experiment import main

EXPERIMENT = "exp4_scoring"
ARMS = [{"name": "autotom_ab", "method": "thoughttracing", "variant": "experiments.exp4_scoring.ab"},
        {"name": "jev_noul", "method": "thoughttracing", "variant": "experiments.exp4_scoring.noul"}]
PLAN = {"mimo": ["paraphrased_tomi"]}

if __name__ == "__main__":
    main(EXPERIMENT, ARMS, PLAN)
