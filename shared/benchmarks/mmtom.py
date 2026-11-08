"""MMToM-QA (text only) in ThoughtTracing's revised, anonymised version: 200 of 600 questions.

ThoughtTracing samples its questions with pandas .sample(n, random_state=123); we freeze the
first 200 of that order. Story, question and options are ThoughtTracing's for both methods.
AutoToM also needs its pre-parsed states and actions, which come from its own CSV with the
original names; those are renamed to ThoughtTracing's letters (see mmtom_rename.py). The two
files hold the same 600 questions in the same order, which load_all checks row by row.
"""

import ast
import csv
import json
import re

from shared.benchmarks.common import EXTERNAL, Example, select_frozen
from shared.benchmarks.mmtom_rename import rename, rename_question_choice


SOURCE = EXTERNAL / "thought-tracing/data/mmtom_qa/revised_questions.jsonl"
AUTOTOM_SOURCE = EXTERNAL / "AutoToM/benchmarks/full_data_formatted/MMToM-QA.csv"
TT_ANSWER_INSTRUCTION = "Please respond with either (a) or (b) without any explanation."


def _split(text):
    context, question = text.split("\nQuestion: ")
    options = re.fullmatch(r"(.*?) \(a\) (.*?) \(b\) (.*?) Please respond with either a or b\.", question)
    if options is None:
        raise ValueError("Unexpected MMToM-QA question layout")
    return context, options[1], (options[2], options[3])


def load_all(source=SOURCE, autotom_source=AUTOTOM_SOURCE):
    rows = [json.loads(line) for line in source.read_text().splitlines() if line.strip()]
    with autotom_source.open() as stream:
        autotom_rows = list(csv.DictReader(stream))
    if len(rows) != len(autotom_rows):
        raise ValueError("ThoughtTracing and AutoToM MMToM-QA files differ in length")
    examples = []
    for index, (row, autotom) in enumerate(zip(rows, autotom_rows)):
        context, question, choices = _split(row["question"])
        answer = choices["ab".index(row["answer"])]
        # Same question in both files: AutoToM's correct option, renamed, is ThoughtTracing's.
        if rename_question_choice(autotom["gt_answer"]) != answer:
            raise ValueError(f"mmtom:{index}: AutoToM row does not match ThoughtTracing row")
        target = re.search(r"Actions taken by (\w+):", context)[1]
        tt_input = row["question"].replace("Please respond with either a or b.", TT_ANSWER_INSTRUCTION)
        examples.append(Example(
            id=f"mmtom:{index}", benchmark="mmtom", context=context, question=question,
            choices=choices, answer=answer, target=target, category=str(row["question_type"]),
            native={"thoughttracing": {"dataset": "mmtom", "input": tt_input + "\nAnswer: "},
                    "autotom": {"dataset_name": "MMToM-QA",
                                "states": [rename(state) for state in ast.literal_eval(autotom["states"])],
                                "actions": [rename(action) for action in ast.literal_eval(autotom["actions"])]}},
        ))
    return examples


def load(source=SOURCE):
    return select_frozen("mmtom", load_all(source), source)
