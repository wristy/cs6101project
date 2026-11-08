import pytest
import json

from shared.benchmarks.paraphrased_tomi import load_examples


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
