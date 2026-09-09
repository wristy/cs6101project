"""Load the released ParaphrasedToMi text and trace files."""

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path
import random
import re


@dataclass(frozen=True)
class Example:
    id: str
    story_id: str
    context: str
    question: str
    choices: tuple[str, ...]
    target: str
    answer: str
    question_type: str
    belief_type: str
    answer_format: str = "free_text"

    def inference_input(self):
        return {key: value for key, value in asdict(self).items()
                if key in {"id", "context", "question", "choices", "target", "answer_format"}}


def load_autotom_examples(path):
    examples = []
    for index, line in enumerate(Path(path).read_text().splitlines()):
        record = json.loads(line)
        question_type = record["question_type"]
        target_match = re.fullmatch(r"Where will (.+?) look for .+\?", record["question"])
        if not question_type.startswith("first_order_") or target_match is None:
            continue
        choices = tuple(record["containers"])
        if len(choices) < 2 or len(set(choices)) != len(choices):
            raise ValueError("AutoToM requires distinct container choices")
        examples.append(Example(
            id=f"autotom:{index}", story_id=sha256(record["story"].encode()).hexdigest(),
            context=record["story"], question=record["question"], choices=choices,
            target=target_match[1], answer=record["answer"], question_type=question_type,
            belief_type="true_belief" if question_type.endswith("_no_tom") else "false_belief",
            answer_format="multiple_choice",
        ))
    return examples


def load_examples(path):
    path = Path(path)
    traces = path.with_suffix(".trace").read_text().splitlines()
    records = []
    sentences = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        if "\t" not in line:
            _, sentence = line.split(" ", 1)
            sentences.append(sentence if sentence.endswith(".") else sentence + ".")
            continue
        question, answer, supporting = line.split("\t")
        int(supporting)
        question = question.split(" ", 1)[1].replace("_", " ")
        choices = tuple(dict.fromkeys(
            sentence.strip(".").rsplit(" ", 1)[-1].replace("_", " ")
            for sentence in sentences
        ))
        context = " ".join(sentences).replace("_", " ")
        records.append((context, question, choices, answer.replace("_", " ")))
        sentences = []
    if sentences or len(records) != len(traces):
        raise ValueError("Dataset and trace counts differ or story is unfinished")
    examples = []
    for index, ((context, question, choices, answer), trace) in enumerate(zip(records, traces)):
        question_type = trace.split(",")[-2]
        target_match = re.fullmatch(r"Where will (.+?) look for .+\?", question)
        if not question_type.startswith("first_order_") or target_match is None:
            continue
        belief_type = "true_belief" if question_type.endswith("_no_tom") else "false_belief"
        examples.append(Example(
            id=f"{path.stem}:{index}", story_id=sha256(context.encode()).hexdigest(),
            context=context, question=question, choices=choices, target=target_match[1],
            answer=answer, question_type=question_type, belief_type=belief_type,
        ))
    return examples


def select_pilot(examples, per_type=10, seed=42):
    if per_type < 1:
        raise ValueError("per_type must be positive")
    shuffled = sorted(examples, key=lambda example: example.id)
    random.Random(seed).shuffle(shuffled)
    used_stories = set()
    debug, pilot = [], []
    for destination, count in ((debug, 1), (pilot, per_type)):
        for belief_type in ("true_belief", "false_belief"):
            candidates = [example for example in shuffled
                          if example.belief_type == belief_type
                          and example.story_id not in used_stories]
            selected = []
            for example in candidates:
                if example.story_id not in used_stories:
                    selected.append(example)
                    used_stories.add(example.story_id)
                if len(selected) == count:
                    break
            if len(selected) != count:
                raise ValueError(f"Not enough distinct stories for {belief_type}")
            destination.extend(selected)
    return debug, pilot