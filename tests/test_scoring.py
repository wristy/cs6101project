"""Scoring follows each benchmark's released rule, made symmetric between the options."""

import pytest

from shared.benchmarks.common import Example
from shared.runner.scoring import score


def item(benchmark, answer, choices):
    return Example(id="x", benchmark=benchmark, context="c", question="q", choices=choices,
                   answer=answer, target="Ava", category="c")


BIG = item("bigtom", "Noor believes it is oat milk.", ("Noor believes it is oat milk.", "Noor believes it is almond milk."))
MM = item("mmtom", "Ava wants an Object H.", ("Ava wants an Object F.", "Ava wants an Object H."))
TOMI = item("paraphrased_tomi", "basket", ("box", "basket"))


@pytest.mark.parametrize("reply,correct,status", [
    ("Answer: a) Noor believes it is oat milk.", True, "ok"), ("b)", False, "ok"), ("a", True, "ok"),
    ("Noor believes it is oat milk.", True, "ok"), ("a) or b)", False, "invalid"), ("unsure", False, "invalid")])
def test_bigtom_option_keys(reply, correct, status):
    graded = score(BIG, "plain", reply)
    assert (graded["correct"], graded["status"]) == (correct, status)


@pytest.mark.parametrize("reply,correct,status", [
    ("(b)", True, "ok"), ("b) Ava wants an Object H.", True, "ok"), ("a.", False, "ok"), ("b", True, "ok"),
    ("(a) and (b)", False, "invalid")])
def test_mmtom_letters(reply, correct, status):
    graded = score(MM, "thoughttracing", reply)
    assert (graded["correct"], graded["status"]) == (correct, status)


def test_tomi_containment_records_other_options():
    assert score(TOMI, "plain", "In the BASKET in the kitchen.")["correct"]
    both = score(TOMI, "plain", "from the box to the basket")
    assert both["correct"] and both["mentions_other_option"]
    assert not score(TOMI, "plain", "In the box.")["correct"]


def test_autotom_must_return_an_option():
    assert score(TOMI, "autotom", "basket")["correct"]
    assert score(TOMI, "autotom", "the basket")["status"] == "invalid"
    assert score(TOMI, "autotom", None)["status"] == "invalid"
