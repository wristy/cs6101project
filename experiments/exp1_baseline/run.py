"""Experiment 1: plain prompting, AutoToM and ThoughtTracing, unchanged.

MiMo runs all three on all four benchmarks; GPT-6 Sol runs all three on BigToM. The MiMo
ThoughtTracing run is the control for Experiments 2-6.

    python -m experiments.exp1_baseline.run --model mimo --benchmark bigtom --arm plain --limit 2
"""

from shared.runner.experiment import BENCHMARKS, main

EXPERIMENT = "exp1_baseline"
ARMS = [{"name": "plain", "method": "plain"},
        {"name": "autotom", "method": "autotom"},
        {"name": "thoughttracing", "method": "thoughttracing"}]
PLAN = {"mimo": list(BENCHMARKS), "gpt6sol": ["bigtom"]}

if __name__ == "__main__":
    main(EXPERIMENT, ARMS, PLAN)
