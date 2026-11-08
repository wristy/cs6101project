"""BigToM as released with ThoughtTracing: 249 questions with at least 90% human agreement.

Each question keeps ThoughtTracing's fixed a/b option order. The condition in task_type
(e.g. "0_forward_action_true_belief") also gives AutoToM's dataset name, which switches on its
BigToM-specific prompts (e.g. "BigToM_fatb").
"""

import json
import re

from shared.benchmarks.common import EXTERNAL, Example, select_frozen


SOURCE = EXTERNAL / "thought-tracing/data/bigtom/bigtom_agree90.json"
AUTOTOM_NAMES = {
    "forward_belief_false_belief": "BigToM_fbfb", "forward_belief_true_belief": "BigToM_fbtb",
    "forward_action_false_belief": "BigToM_fafb", "forward_action_true_belief": "BigToM_fatb",
    "backward_belief_false_belief": "BigToM_bbfb", "backward_belief_true_belief": "BigToM_bbtb",
}
# ThoughtTracing's BigToM prompt (bigtom/test_bigtom.py), used as the tracer input and baseline.
TT_INSTRUCTION = ("Answer the questions based on the context. Keep your answer concise, few words are "
                  "enough, maximum one sentence. Answer as 'Answer:<option>)<answer>'.")


def load_all(source=SOURCE):
    examples = []
    for key, row in json.loads(source.read_text()).items():
        lines = row["question"].split("\n")
        if len(lines) != 4 or lines[1] != "Choose one of the following:":
            raise ValueError(f"bigtom:{key}: unexpected question layout")
        choices = tuple(line[2:].strip() for line in lines[2:])
        if not (lines[2].startswith("a)") and lines[3].startswith("b)")):
            raise ValueError(f"bigtom:{key}: options are not a) and b)")
        answer = choices["ab".index(row["true_answer_key"][0])]
        if row["true_answer"][3:].strip() != answer:
            raise ValueError(f"bigtom:{key}: true_answer does not match its key")
        condition = row["task_type"].split("_", 1)[1]
        match = re.match(r"(?:Does (\w+) believe|What will (\w+) do)", row["original_question"])
        if match is None:
            raise ValueError(f"bigtom:{key}: cannot find the target agent")
        tt_input = (f"{TT_INSTRUCTION}\n\nStory: {row['context']}\nQuestion: {row['question']}\nAnswer:")
        examples.append(Example(
            id=f"bigtom:{key}", benchmark="bigtom", context=row["context"],
            question=row["original_question"], choices=choices, answer=answer,
            target=match[1] or match[2], category=condition,
            belief_type="false_belief" if condition.endswith("false_belief") else "true_belief",
            native={"thoughttracing": {"dataset": "bigtom", "input": tt_input},
                    "autotom": {"dataset_name": AUTOTOM_NAMES[condition]}},
        ))
    return examples


def load(source=SOURCE):
    return select_frozen("bigtom", load_all(source), source)
