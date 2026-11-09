"""The shared + experiments layout is in place and the old pilot scaffolding is gone."""

import importlib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

SHARED_MODULES = [
    "shared", "shared.bridge", "shared.models", "shared.models.client", "shared.methods",
    "shared.methods.worker", "shared.methods.runner", "shared.benchmarks",
    "shared.benchmarks.paraphrased_tomi", "shared.benchmarks.bigtom", "shared.benchmarks.mmtom",
    "shared.benchmarks.exploretom", "shared.methods.prompts", "shared.runner.scoring",
    "shared.runner.experiment", "experiments.exp1_baseline.run", "experiments.exp3_persona.run",
    "experiments.exp3_persona.variant", "experiments.exp4_scoring.run", "experiments.exp4_scoring.ab",
    "experiments.exp4_scoring.noul", "shared.models.typesafe", "shared.tracing.structured",
    "shared.tracing.format_check", "experiments.exp2a_mental_models.run", "experiments.exp2a_mental_models.variant",
    "experiments.exp2b_triples.run", "experiments.exp2b_triples.variant", "experiments.exp5_rejuvenation.run",
    "experiments.exp5_rejuvenation.variant", "experiments.exp6_summary.run", "experiments.exp6_summary.variant", "shared.runner", "shared.runner.evaluation",
    "shared.tracing", "shared.analysis", "shared.analysis.stats", "shared.analysis.aggregate", "shared.runner.pilot", "experiments.exp5_rejuvenation.calibrate", "experiments.exp2a_mental_models.check_choice", "experiments",
]


@pytest.mark.parametrize("module", SHARED_MODULES)
def test_shared_modules_import(module):
    importlib.import_module(module)


def test_layout_folders_exist():
    for folder in ["shared/benchmarks/data", "experiments", "results/aggregate", "patches"]:
        assert (ROOT / folder).is_dir(), folder


def test_pilot_scaffolding_removed():
    for path in ["src", "notebooks", "configs", "data", "docs", "probes"]:
        assert not (ROOT / path).exists(), path


def test_patches_route_to_shared_bridge():
    for patch in (ROOT / "patches").glob("*.patch"):
        text = patch.read_text()
        assert "tom_research" not in text
        assert "shared.bridge" in text


def test_env_example_names_every_key_and_env_is_ignored():
    example = (ROOT / ".env.example").read_text()
    for key in ["DEEPINFRA_API_KEY", "OPENAI_API_KEY", "JEV_API_KEY"]:
        assert key in example
    assert ".env" in (ROOT / ".gitignore").read_text().splitlines()


@pytest.mark.live
def test_live_harness_reads_keys_from_env(require_key):
    """Runs only with --live. Proves .env is loaded; makes no API call."""
    assert require_key("DEEPINFRA_API_KEY")
