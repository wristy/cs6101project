"""Experiment 3: observer persona in ThoughtTracing's propagation step, on ParaphrasedToMi.

The control is Experiment 1's MiMo ThoughtTracing run on the same questions.

    python -m experiments.exp3_persona.run --limit 2
"""

from shared.runner.experiment import main

EXPERIMENT = "exp3_persona"
ARMS = [{"name": "observer", "method": "thoughttracing", "variant": "experiments.exp3_persona.variant"}]
PLAN = {"mimo": ["paraphrased_tomi"]}

if __name__ == "__main__":
    main(EXPERIMENT, ARMS, PLAN)
