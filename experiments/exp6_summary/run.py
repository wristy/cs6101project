"""Experiment 6: running perception summary for propagation and weighting, on MMToM-QA.

The control is Experiment 1's MiMo ThoughtTracing run.

    python -m experiments.exp6_summary.run --limit 2
"""

from shared.runner.experiment import main

EXPERIMENT = "exp6_summary"
ARMS = [{"name": "summary", "method": "thoughttracing", "variant": "experiments.exp6_summary.variant"}]
PLAN = {"mimo": ["mmtom"]}

if __name__ == "__main__":
    main(EXPERIMENT, ARMS, PLAN)
