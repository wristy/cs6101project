"""Step 2 live checks: 7 small real API calls in total, plus 2 Jev calls. Run with `pytest --live`.

Each test skips if its key is missing from .env. Call logs go to results/live-tests/ (git-ignored).
"""

from pathlib import Path

import pytest

from shared.models.client import LLMClient
from shared.models.embeddings import EmbeddingClient, cosine
from shared.models.providers import model_config
from shared.models.typesafe import JevClient

pytestmark = pytest.mark.live
LOGS = Path(__file__).resolve().parents[1] / "results/live-tests"

CLEAR_LIKELY = """Determine if the statement is likely, respond with only either A or B.
Sally put the apple in the basket and left. Nobody touched the basket.
Determine if the following statement is likely: Sally believes the apple is in the basket.
A) Likely.
B) Unlikely."""
CLEAR_UNLIKELY = CLEAR_LIKELY.replace("is in the basket.\nA)", "is on the moon.\nA)")


@pytest.mark.parametrize("name,key", [("mimo", "DEEPINFRA_API_KEY"), ("gpt6sol", "OPENAI_API_KEY")])
def test_generate_returns_text_without_thinking(name, key, require_key):
    require_key(key)
    client = LLMClient(model_config(name, max_calls=1), log_path=LOGS / f"{name}.jsonl")
    text = client.generate("Answer in one word: what colour is a clear daytime sky?", max_tokens=32)
    assert "blue" in text.lower()
    assert client.usage()["calls"] == 1


@pytest.mark.parametrize("name,key", [("mimo", "DEEPINFRA_API_KEY"), ("gpt6sol", "OPENAI_API_KEY")])
def test_ab_rule_separates_likely_from_unlikely(name, key, require_key):
    require_key(key)
    client = LLMClient(model_config(name, max_calls=2), log_path=LOGS / f"{name}.jsonl")
    likely, unlikely = client.score_label(CLEAR_LIKELY), client.score_label(CLEAR_UNLIKELY)
    assert likely > 0.5 > unlikely, (likely, unlikely)


def test_embeddings_rank_a_paraphrase_above_an_unrelated_sentence(require_key):
    require_key("DEEPINFRA_API_KEY")
    a, paraphrase, unrelated = EmbeddingClient(max_calls=1, log_path=LOGS / "embeddings.jsonl").embed([
        "Sally thinks the apple is still in the basket.",
        "Sally believes the apple remains in the basket.",
        "The stock market closed higher on Friday.",
    ])
    assert cosine(a, paraphrase) > cosine(a, unrelated)


def test_jev_noul_separates_true_from_false(require_key):
    require_key("JEV_API_KEY")
    client = JevClient(max_calls=2, log_path=LOGS / "jev.jsonl")
    state = "Sally put the apple in the basket and left. Nobody touched the basket."
    likely = client.noul(state, "Sally believes the apple is in the basket.")
    unlikely = client.noul(state, "Sally believes the apple is on the moon.")
    assert likely > 0.5 > unlikely, (likely, unlikely)
