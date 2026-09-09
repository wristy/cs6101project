import pytest
import json

from tom_research.data import load_examples, select_pilot


def test_autotom_uses_released_containers_and_question_belief_type(tmp_path):
    from tom_research.data import load_autotom_examples

    source = tmp_path / "test_balanced.jsonl"
    record = {
        "story": "1 Ava entered the kitchen.\n2 The apple is in the basket.",
        "question": "Where will Ava look for the apple?", "answer": "basket",
        "containers": ["box", "basket"], "question_type": "first_order_0_no_tom",
        "story_type": "second_order_false_belief",
    }
    source.write_text(json.dumps(record) + "\n")
    example = load_autotom_examples(source)[0]
    assert example.id == "autotom:0"
    assert example.choices == ("box", "basket")
    assert example.belief_type == "true_belief"
    assert example.answer_format == "multiple_choice"
    assert example.context == record["story"]
    record["answer"] = "box"
    source.write_text(json.dumps(record) + "\n")
    assert load_autotom_examples(source)[0].inference_input() == example.inference_input()


def write_dataset(tmp_path, trace=None):
    source = tmp_path / "test.txt"
    source.write_text(
        "1 Ava entered the kitchen.\n"
        "2 The apple is in the wooden_box.\n"
        "3 Ava moved the apple to the basket.\n"
        "4 Where will Ava look for the apple?\tbasket\t1\n"
    )
    source.with_suffix(".trace").write_text(
        trace or "enter_agent_0,first_order_0_no_tom,second_order_false_belief\n"
    )
    return source


def test_load_preserves_metadata_and_derives_candidates_without_gold(tmp_path):
    source = write_dataset(tmp_path)
    example = load_examples(source)[0]
    assert example.id == "test:0"
    assert example.belief_type == "true_belief"
    assert example.target == "Ava"
    assert example.choices == ("kitchen", "wooden box", "basket")
    assert example.answer == "basket"
    assert "answer" not in example.inference_input()
    source.write_text(source.read_text().replace("\tbasket\t", "\tkitchen\t"))
    assert load_examples(source)[0].inference_input() == example.inference_input()


def test_rejects_mismatched_trace_count(tmp_path):
    source = write_dataset(tmp_path)
    source.with_suffix(".trace").write_text("")
    with pytest.raises(ValueError, match="trace"):
        load_examples(source)


def test_select_pilot_is_balanced_stable_and_story_disjoint(tmp_path):
    from dataclasses import replace

    base = load_examples(write_dataset(tmp_path))[0]
    examples = [
        replace(base, id=f"item:{index}", story_id=f"story:{index // 2}",
                belief_type="true_belief" if index % 2 else "false_belief")
        for index in range(24)
    ]
    debug, pilot = select_pilot(examples, per_type=3, seed=42)
    assert len(debug) == 2
    assert len(pilot) == 6
    assert sum(item.belief_type == "true_belief" for item in pilot) == 3
    assert not {item.story_id for item in debug} & {item.story_id for item in pilot}
    assert len({item.story_id for item in pilot}) == len(pilot)
    assert (debug, pilot) == select_pilot(examples, per_type=3, seed=42)


def test_insufficient_examples_raise(tmp_path):
    with pytest.raises(ValueError, match="enough"):
        select_pilot(load_examples(write_dataset(tmp_path)), per_type=10)


def test_selection_does_not_depend_on_gold_candidate_coverage(tmp_path):
    from dataclasses import replace

    base = load_examples(write_dataset(tmp_path))[0]
    examples = [
        replace(base, id=str(index), story_id=str(index), answer="not in candidates",
                belief_type="true_belief" if index % 2 else "false_belief")
        for index in range(8)
    ]
    debug, pilot = select_pilot(examples, per_type=2)
    assert len(debug) == 2 and len(pilot) == 4