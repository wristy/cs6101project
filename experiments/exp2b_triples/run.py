"""Experiment 2b: structured hypotheses (see variant.py), on paraphrased_tomi.

The control is Experiment 1's MiMo ThoughtTracing run on the same questions.

    python -m experiments.exp2b_triples.run --limit 2
"""

from shared.runner.experiment import main

EXPERIMENT = "exp2b_triples"
ARMS = [{"name": "triples", "method": "thoughttracing", "variant": "experiments.exp2b_triples.variant"}]
PLAN = {"mimo": ["paraphrased_tomi"]}

if __name__ == "__main__":
    main(EXPERIMENT, ARMS, PLAN)
