"""Experiments 2a/2b: schemas, rendering, Jev's model choice, and the structured tracer steps."""

import json
import subprocess
import sys

import pytest

from experiments.exp2a_mental_models import run as run2a, variant as v2a
from experiments.exp2b_triples import run as run2b, variant as v2b
from shared.models.typesafe import JevClient
from shared.tracing.structured import StructureError, list_schema, parse_json, validate
from tests.upstream_source import ROOT, TRACER

NESTED = {"triples": [{"subject": "Sally", "relation": "IsUnawareOf", "object": "the apple moved"},
                      {"subject": "Anne", "relation": "Believes", "object": {
                          "subject": "Sally", "relation": "Believes",
                          "object": {"subject": "apple", "relation": "LocatedIn", "object": "basket"}}}]}


def test_triples_render_and_validate():
    validate(NESTED, v2b.SCHEMA)
    assert v2b.render("Sally", NESTED) == (
        "Sally is unaware of the apple moved. Anne believes that Sally believes that apple is in basket.")


@pytest.mark.parametrize("bad", [
    {"triples": [{"subject": "apple", "relation": "LocatedIn", "object": "box"}]},     # LocatedIn at top level
    {"triples": [{"subject": "Sally", "relation": "Thinks", "object": "x"}]},           # unknown relation
    {"triples": []}, {"triples": [NESTED["triples"][0]] * 5},                           # 0 or too many
    {"triples": [{"subject": "Sally", "relation": "Sees"}]}])                          # missing object
def test_triples_reject_bad_structure(bad):
    with pytest.raises(StructureError):
        validate(bad, v2b.SCHEMA)


def test_mental_models_render_their_slots():
    belief = v2a.MODELS[0]
    assert belief.render("Ava", {"observation": "the plate in the fridge.", "belief": "it is still there"}) == (
        "Ava observed the plate in the fridge, so Ava believes it is still there.")
    with pytest.raises(StructureError):
        validate({"observation": "x"}, belief.schema)
    assert [m.name for m in v2a.MODELS] == ["belief", "goal", "belief_goal", "higher_order"]


def test_jev_choice_uses_the_story_and_takes_the_most_probable_model():
    sent = []

    def transport(request):
        sent.append(request)
        return {"answers": {"q": {"type": "choice", "choice": "belief_goal", "confidence": 0.5,
                                  "probabilities": {"belief": 0.1, "goal": 0.2, "belief_goal": 0.6,
                                                    "higher_order": 0.1}}}}
    model, record = v2a.choose_model(JevClient(transport=transport), "Ava walks to the kitchen.")
    assert model.name == "belief_goal" and record["probabilities"]["belief_goal"] == 0.6
    assert sent[0]["state"] == "Ava walks to the kitchen."
    assert sent[0]["questions"]["q"]["type"] == "choice"
    assert sent[0]["questions"]["q"]["criteria"] == {m.name: m.description for m in v2a.MODELS}
    assert record["jev_choice"] == "belief_goal"


def test_list_schema_and_json_parsing():
    schema = list_schema(v2b.SCHEMA, 2)
    validate({"hypotheses": [NESTED, NESTED]}, schema)
    with pytest.raises(StructureError):
        validate({"hypotheses": [NESTED]}, schema)
    assert parse_json('```json\n{"a": 1}\n```') == {"a": 1}
    with pytest.raises(StructureError):
        parse_json("not json")


def test_plans():
    assert run2a.PLAN == {"mimo": ["mmtom"]} and run2b.PLAN == {"mimo": ["paraphrased_tomi"]}


SCRIPT = r'''
import json
from shared import bridge
from shared.models.client import LLMClient, ModelConfig
from shared.methods.worker import load_upstream, load_variant

triple = {"triples": [{"subject": "Ava", "relation": "Believes",
                       "object": {"subject": "apple", "relation": "LocatedIn", "object": "basket"}}]}
replies = iter(["not json", json.dumps({"hypotheses": [triple] * 5})] + [json.dumps(triple)] * 4)
requests = []
def transport(request):
    requests.append(request)
    return {"choices": [{"message": {"content": next(replies)}, "finish_reason": "stop"}], "usage": {}}
bridge.configure(LLMClient(ModelConfig(), transport=transport))
upstream = load_upstream("thoughttracing")
cls = load_variant("experiments.exp2b_triples.variant", upstream)
tracer = cls.__new__(cls)
tracer.target_agent, tracer.assumption, tracer.trace_header = "Ava", "", "HEADER"
tracer.args = type("A", (), {"n_hypotheses": 4})()
tracer.get_agent_state = lambda agent, state: state
initial = tracer.initialize({"state": "Ava is home.", "action": None}, {"state": "Ava sees home.", "action": None})
assert initial.texts == ["Ava believes that apple is in basket."] * 4, initial.texts
assert requests[0].get("response_format") is None     # off: MiMo passed the format check without it
assert requests[1]["temperature"] == 0.33           # retried after invalid JSON, as upstream
assert requests[1]["response_format"]["json_schema"]["strict"] is True   # enforced once broken
tracer.setup_propagation = lambda h, s, p: {"target_agent": "Ava", "context_history": [], "perception_history": [],
                                            "context_and_perception_str": "past", "new_context": "now"}
updated = tracer.propagate(initial, {}, {})
assert updated.texts == ["Ava believes that apple is in basket."] * 4
last = requests[-1]["messages"]
assert last[0]["content"] == "You are an expert assistant trying to predict Ava's thoughts."
assert json.dumps(triple) in last[-1]["content"]      # previous hypothesis shown in its structured form
assert [entry["ok"] for entry in tracer.structure_log] == [False, True, True, True, True, True]
'''


@pytest.mark.skipif(not TRACER.exists(), reason="Run scripts/bootstrap.sh")
def test_structured_tracer_initializes_and_propagates_real_hypothesis_sets():
    result = subprocess.run([sys.executable, "-c", SCRIPT], cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout[-3000:] + result.stderr[-3000:]


def test_nested_objects_render_as_clauses():
    hypothesis = {"triples": [{"subject": "Owen", "relation": "IsUnawareOf",
                               "object": {"subject": "Noah", "relation": "Moved", "object": "peach"}}]}
    assert v2b.render("Owen", hypothesis) == "Owen is unaware that Noah moved peach."


def test_choice_check_sends_distinct_story_texts_without_questions():
    from experiments.exp2a_mental_models import check_choice
    texts = check_choice.stories(5)
    assert len(texts) == len(set(texts)) >= 2
    assert all("Question:" not in text for text in texts)


def test_overlong_lists_keep_their_first_items_like_upstream():
    from experiments.exp2b_triples.variant import SCHEMA
    from shared.tracing.structured import list_schema, trim, validate
    triple = {"subject": "Ava", "relation": "Believes",
              "object": {"subject": "apple", "relation": "LocatedIn", "object": "basket"}}
    hypothesis = {"triples": [triple] * 6}
    assert len(validate(trim(hypothesis, SCHEMA), SCHEMA)["triples"]) == 4
    many = {"hypotheses": [hypothesis] * 5}
    trimmed = validate(trim(many, list_schema(SCHEMA, 4)), list_schema(SCHEMA, 4))
    assert len(trimmed["hypotheses"]) == 4 and all(len(h["triples"]) == 4 for h in trimmed["hypotheses"])
    with pytest.raises(StructureError):        # a bare world fact at the top level is still refused
        validate(trim({"triples": [{"subject": "peach", "relation": "LocatedIn", "object": "box"}]}, SCHEMA), SCHEMA)
