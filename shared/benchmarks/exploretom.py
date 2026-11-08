"""ExploreToM (Meta, CC BY-NC 4.0): 200 location-belief questions on its infilled stories.

ExploreToM has no ThoughtTracing or AutoToM release, so we pick questions the same way as
ParaphrasedToMi: "In which container will X search..." (first order) and "...does X think that Y
will search..." (second order), 50 each of first/second order x false/true belief. The release
marks a question "-True" when the belief matches where the object really is and "-False" when
it does not. We use the natural-language infilled story, only questions whose answer appears word for word
in it, and only options that do. Each cell takes one question per story first, then tops
up from stories already used: the false-belief cells use 46 and 48 stories. Only 50 first-order
false-belief questions qualify, so that cell uses all of them.

Options are the containers the object is known to have been in: the believed one, where it
really is now and where it started (the release's ground_truth and memory questions), in order
of first mention in the story. Both methods use their ToMi settings, the closest released ones.
The data file is downloaded by hand (see data/exploretom/README.md) and checked by sha256.
"""

import ast
import re

from shared.benchmarks.common import FROZEN, Example, file_sha256, freeze, select_frozen


SOURCE = FROZEN / "exploretom" / "ExploreToM-data-sample.csv"
SOURCE_SHA256 = "f8462b1a2199819a2baef5ab321b7b9240249fe229b94c86d8d06c2c47fc448b"
SOURCE_URL = ("https://huggingface.co/datasets/facebook/ExploreToM/resolve/"
              "a4ac6f257e0034945f829716047ae6306dc625a0/ExploreToM-data-sample.csv")
PER_CELL = 50
CELLS = tuple(f"{order}_order_{belief}" for order in ("first", "second")
              for belief in ("false_belief", "true_belief"))
QUESTION_PATTERNS = (re.compile(r"In which container will (.+?) search for the .+\?"),
                     re.compile(r"In which container does (.+?) think that .+? will search for the .+\?"))


def _read(source):
    import pandas as pd
    if file_sha256(source) != SOURCE_SHA256:
        raise ValueError(f"{source} is not the pinned ExploreToM release file; download {SOURCE_URL}")
    data = pd.read_csv(source)
    params = data["qprop=params"].map(ast.literal_eval)
    data["kind"], data["object"] = params.map(lambda p: p[2]), params.map(lambda p: p[1])
    return data


def load_all(source=SOURCE):
    """Every usable location-belief question, with ids "exploretom:<row number>"."""
    data = _read(source)
    facts = {kind: data[data["kind"] == f"{kind}-container_location"]
             .groupby(["infilled_story", "object"])["expected_answer"].first()
             for kind in ("ground_truth", "memory")}
    examples = []
    for row_number, row in data[data["kind"].str.startswith("container_location-")].iterrows():
        order = {1: "first", 2: "second"}[row["qprop=nth_order"]]
        belief = "true_belief" if row["kind"].endswith("-True") else "false_belief"
        story, key = row["infilled_story"], (row["infilled_story"], row["object"])
        known = [row["expected_answer"]] + [facts[kind][key] for kind in facts if key in facts[kind]]
        lowered = story.lower()
        if row["expected_answer"].lower() not in lowered:
            continue
        named = [option for option in dict.fromkeys(known) if option.lower() in lowered]
        choices = tuple(sorted(named, key=lambda option: lowered.index(option.lower())))
        if len(choices) < 2:
            continue
        match = next((m for p in QUESTION_PATTERNS if (m := p.fullmatch(row["question"]))), None)
        if match is None:
            raise ValueError(f"exploretom:{row_number}: unexpected question {row['question']!r}")
        examples.append(Example(
            id=f"exploretom:{row_number}", benchmark="exploretom", context=story,
            question=row["question"], choices=choices, answer=row["expected_answer"],
            target=match[1], category=f"{order}_order_{belief}", belief_type=belief,
            answer_format="free_text", story_id=str(row["sprop=global_idx"]),
            native={"thoughttracing": {"dataset": "tomi",
                                       "input": f"{story}\n\nQuestion: {row['question']}\nAnswer:"},
                    "autotom": {"dataset_name": "ToMi-1st" if order == "first" else "ToMi-2nd"}},
        ))
    return examples


def freeze_selection(source=SOURCE, seed=123):
    """Shuffle with a fixed seed; fill each cell with one question per story, then top it up."""
    import numpy as np
    examples = load_all(source)
    order = np.random.RandomState(seed).permutation(len(examples))
    chosen, used = {cell: [] for cell in CELLS}, {cell: set() for cell in CELLS}
    for repeat_stories in (False, True):
        for index in order:
            example = examples[index]
            cell = example.category
            if len(chosen[cell]) >= PER_CELL or example in chosen[cell]:
                continue
            if repeat_stories or example.story_id not in used[cell]:
                chosen[cell].append(example)
                used[cell].add(example.story_id)
    short = {cell: len(items) for cell, items in chosen.items() if len(items) < PER_CELL}
    if short:
        raise ValueError(f"Not enough questions: {short}")
    return freeze("exploretom", [e for cell in CELLS for e in chosen[cell]], source,
                  "Location-belief questions, 50 per order x belief cell, one per story per cell first; "
                  "numpy RandomState(123).permutation over load_all() order.")


def load(source=SOURCE):
    return select_frozen("exploretom", load_all(source), source)
