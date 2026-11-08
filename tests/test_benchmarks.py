"""The frozen question sets: counts, balance, no answer leaks, and reproducible freezing."""

from collections import Counter
import json

import pytest

from shared.benchmarks import bigtom, common, mmtom, paraphrased_tomi
from shared.benchmarks.mmtom_rename import rename, rename_question_choice

pytestmark = pytest.mark.skipif(not bigtom.SOURCE.exists(),
                                reason="Run scripts/bootstrap.sh to fetch pinned sources")
LOADERS = {"bigtom": bigtom, "mmtom": mmtom, "paraphrased_tomi": paraphrased_tomi}


def test_bigtom_is_all_249_with_autotom_condition_names():
    examples = bigtom.load()
    assert len(examples) == 249
    assert Counter(e.native["autotom"]["dataset_name"] for e in examples) == {
        "BigToM_fbfb": 85, "BigToM_fatb": 70, "BigToM_bbfb": 35, "BigToM_fbtb": 24,
        "BigToM_fafb": 22, "BigToM_bbtb": 13}
    assert all(e.target in e.question for e in examples)


def test_mmtom_is_thoughttracings_200_question_sample():
    examples = mmtom.load()
    assert len(examples) == 200 and len({e.id for e in examples}) == 200
    assert set(Counter(e.category for e in examples)) == {"1.1", "1.2", "1.3", "2.1", "2.2", "2.3", "2.4"}
    assert all(e.native["autotom"]["states"] and e.native["autotom"]["actions"] for e in examples)


def test_paraphrased_tomi_is_balanced_by_order_and_belief():
    examples = paraphrased_tomi.load()
    assert len(examples) == 200
    assert Counter(e.category for e in examples) == {t: 25 for t in paraphrased_tomi.TYPES}
    assert Counter((e.category.split("_")[0], e.belief_type) for e in examples) == {
        ("first", "false_belief"): 50, ("first", "true_belief"): 50,
        ("second", "false_belief"): 50, ("second", "true_belief"): 50}


@pytest.mark.parametrize("name", LOADERS)
def test_questions_are_well_formed_and_never_leak_the_answer(name):
    for example in LOADERS[name].load():
        visible = example.inference_input()
        assert "answer" not in visible and "category" not in visible
        assert example.target and example.context and example.question
        assert len(example.choices) >= 2 and len(set(example.choices)) == len(example.choices)
        assert example.answer in example.choices
        assert set(example.native) == {"thoughttracing", "autotom"}
        keys = {key for method in example.native.values() for key in method}
        assert not keys & {"answer", "gt_answer", "true_answer", "label"}
        tt_input = example.native["thoughttracing"]["input"]
        assert tt_input.rstrip().endswith("Answer:"), "the tracer input must stop before the answer"


@pytest.mark.parametrize("name", LOADERS)
def test_freezing_again_gives_the_committed_list(name, tmp_path, monkeypatch):
    from scripts import freeze_benchmarks
    committed = json.loads((common.FROZEN / f"{name}.json").read_text())
    monkeypatch.setattr(common, "FROZEN", tmp_path)
    path = freeze_benchmarks.FREEZERS[name]()
    assert json.loads(path.read_text()) == committed


def test_changed_source_file_is_rejected(tmp_path):
    copy = tmp_path / "bigtom.json"
    copy.write_text(bigtom.SOURCE.read_text().replace("Noor", "Nora", 1))
    with pytest.raises(ValueError, match="differs from the frozen one"):
        bigtom.load(copy)


def test_mmtom_rename_matches_thoughttracing_names():
    assert rename("Mary walks to the kitchen and opens the refrigerator.") == \
        "Mary walks to the Room C and opens the fridge."
    assert rename("A plate and two wine glasses, a bag of chips and an apple.") == \
        "An Object H and two Object Ls, an Object B and an Object A."
    assert rename("the second kitchen cabinet in the living room") == "the second Room C cabinet in the Room D"
    assert rename_question_choice("inside the first kitchen cabinet") == "inside the 1st Room C cabinet"


def test_mmtom_autotom_inputs_use_only_renamed_names():
    original = r"\b(refrigerator|kitchen|bedroom|bathroom|living room|apple|chips|book|wine|condiment|" \
               r"cupcake|dish bowl|plate|remote|salmon|water glass)\b"
    import re
    for example in mmtom.load():
        text = " ".join(example.native["autotom"]["states"] + example.native["autotom"]["actions"])
        assert not re.search(original, text), (example.id, re.search(original, text))
