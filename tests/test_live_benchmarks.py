"""Step 3 live check: up to 4 MiMo calls, one frozen question per benchmark. Run with `pytest --live`.

This only checks that a loaded question reaches the model intact and gets a usable answer back;
accuracy is measured by the experiments, not here.
"""

from pathlib import Path

import pytest

from shared.benchmarks import bigtom, exploretom, mmtom, paraphrased_tomi
from shared.models.client import LLMClient
from shared.models.providers import model_config

pytestmark = [pytest.mark.live, pytest.mark.skipif(not bigtom.SOURCE.exists(), reason="run bootstrap")]
LOGS = Path(__file__).resolve().parents[1] / "results/live-tests"


@pytest.mark.parametrize("loader", [bigtom, mmtom, paraphrased_tomi, exploretom],
                         ids=lambda m: m.__name__.split(".")[-1])
def test_first_frozen_question_gets_an_answer_from_its_options(loader, require_key):
    require_key("DEEPINFRA_API_KEY")
    if not loader.SOURCE.exists():
        pytest.skip("data file not downloaded")
    example = loader.load()[0]
    client = LLMClient(model_config("mimo", max_calls=1), log_path=LOGS / "benchmarks.jsonl")
    options = "\n".join(f"- {choice}" for choice in example.choices)
    answer = client.generate(f"{example.context}\n\nQuestion: {example.question}\n\nOptions:\n{options}\n\n"
                             "Reply with exactly one option as written, without explanation.", max_tokens=128)
    assert any(choice.lower().strip(".") in answer.lower() for choice in example.choices), answer
