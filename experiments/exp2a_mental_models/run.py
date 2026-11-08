"""Experiment 2a: structured hypotheses (see variant.py), on mmtom.

The control is Experiment 1's MiMo ThoughtTracing run on the same questions.

    python -m experiments.exp2a_mental_models.run --limit 2
"""

from shared.runner.experiment import main

EXPERIMENT = "exp2a_mental_models"
ARMS = [{"name": "mental_models", "method": "thoughttracing", "variant": "experiments.exp2a_mental_models.variant"}]
PLAN = {"mimo": ["mmtom"]}

if __name__ == "__main__":
    main(EXPERIMENT, ARMS, PLAN)
