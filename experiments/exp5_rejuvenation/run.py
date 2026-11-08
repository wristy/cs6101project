"""Experiment 5: embedding-based duplicate replacement in ThoughtTracing, on MMToM-QA.

The control is Experiment 1's MiMo ThoughtTracing run. Set EXP5_THRESHOLD to the calibrated
cosine threshold before the full run (default 0.85).

    python -m experiments.exp5_rejuvenation.run --limit 2
"""

from shared.runner.experiment import main

EXPERIMENT = "exp5_rejuvenation"
ARMS = [{"name": "embedding_rejuvenation", "method": "thoughttracing",
         "variant": "experiments.exp5_rejuvenation.variant"}]
PLAN = {"mimo": ["mmtom"]}

if __name__ == "__main__":
    main(EXPERIMENT, ARMS, PLAN)
