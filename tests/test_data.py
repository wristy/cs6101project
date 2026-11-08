"""ParaphrasedToMi parsing on a tiny hand-written file."""

import pytest

from shared.benchmarks.paraphrased_tomi import load_all


def write_dataset(tmp_path, question="Where will Ava look for the apple?", trace=None):
    source = tmp_path / "test.txt"
    source.write_text(
        "1 Ava entered the kitchen.\n"
        "2 The apple is in the wooden_box.\n"
        "3 Ava moved the apple to the basket.\n"
        f"4 {question}\tbasket\t1\n"
    )
    source.with_suffix(".trace").write_text(
        trace or "enter_agent_0,first_order_0_no_tom,second_order_false_belief\n"
    )
    return source


def test_load_preserves_metadata_and_derives_candidates_without_gold(tmp_path):
    source = write_dataset(tmp_path)
    example = load_all(source)[0]
    assert example.id == "test:0"
    assert example.belief_type == "true_belief"
    assert example.target == "Ava"
    assert example.choices == ("kitchen", "wooden box", "basket")
    assert example.answer == "basket"
    assert example.answer_format == "free_text"
    assert example.native["thoughttracing"]["input"].endswith("Question: Where will Ava look for the apple?\nAnswer:")
    assert example.native["autotom"]["dataset_name"] == "ToMi-1st"
    assert "answer" not in example.inference_input()
    source.write_text(source.read_text().replace("\tbasket\t", "\tkitchen\t"))
    assert load_all(source)[0].inference_input() == example.inference_input()


def test_second_order_target_is_the_outer_agent(tmp_path):
    source = write_dataset(tmp_path, "Where does Ben think that Ava searches for the apple?",
                           "enter_agent_0,second_order_1_tom,second_order_false_belief\n")
    example = load_all(source)[0]
    assert example.target == "Ben"
    assert example.belief_type == "false_belief"
    assert example.native["autotom"]["dataset_name"] == "ToMi-2nd"


def test_skips_reality_and_memory_questions(tmp_path):
    source = write_dataset(tmp_path, "Where is the apple really?", "enter_agent_0,reality,x\n")
    assert load_all(source) == []


def test_rejects_unknown_question_shape(tmp_path):
    source = write_dataset(tmp_path, "Why did Ava move the apple?")
    with pytest.raises(ValueError, match="unexpected question"):
        load_all(source)


def test_rejects_mismatched_trace_count(tmp_path):
    source = write_dataset(tmp_path)
    source.with_suffix(".trace").write_text("")
    with pytest.raises(ValueError, match="trace"):
        load_all(source)
