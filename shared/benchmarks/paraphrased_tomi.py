"""ParaphrasedToMi as released with ThoughtTracing: 200 first- and second-order questions.

ThoughtTracing shuffles the test split with pandas .sample(n=2000, random_state=123) and takes the
first N questions of each type in that order (revised_tomi/test_tomi.py). We do the same with
N = 25 for each of the eight first/second-order types, which gives 100 first-order and 100
second-order questions, each split 50 false belief / 50 true belief. Answers are free text.

Options (used by AutoToM and the multiple-choice baseline) are the ToMi containers the story
names, in order of first mention: two in 96% of the 4,004 questions, three in the rest. The
correct answer is always among them. ThoughtTracing's own candidate list (the last word of each
sentence) does not work on the paraphrased stories and missed the answer in 32 of our 200.
"""

from hashlib import sha256
import re

from shared.benchmarks.common import EXTERNAL, Example, select_frozen


SOURCE = EXTERNAL / "thought-tracing/data/paraphrased_tomi/test.txt"
TYPES = tuple(f"{order}_order_{variant}" for order in ("first", "second")
              for variant in ("0_tom", "1_tom", "0_no_tom", "1_no_tom"))
# Every answer to a first- or second-order question in the test split is one of these.
CONTAINERS = ("treasure chest", "container", "envelope", "cupboard", "suitcase", "bathtub", "pantry",
              "bucket", "bottle", "drawer", "basket", "crate", "box")
_CONTAINER = re.compile(r"\b(" + "|".join(re.escape(c) for c in CONTAINERS) + r")\b")
QUESTION_PATTERNS = (re.compile(r"Where will (\w+) look for .+\?"),
                     re.compile(r"Where does (\w+) think that \w+ searches for .+\?"))


def _records(path):
    traces = path.with_suffix(".trace").read_text().splitlines()
    records, sentences = [], []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        if "\t" not in line:
            _, sentence = line.split(" ", 1)
            sentences.append(sentence if sentence.endswith(".") else sentence + ".")
            continue
        question, answer, supporting = line.split("\t")
        int(supporting)
        records.append((sentences, question.split(" ", 1)[1], answer.replace("_", " ")))
        sentences = []
    if sentences or len(records) != len(traces):
        raise ValueError("Dataset and trace counts differ or story is unfinished")
    return records, traces


def load_all(source=SOURCE):
    """Every first- and second-order question, with ids "<split>:<question index>"."""
    records, traces = _records(source)
    examples = []
    for index, ((sentences, raw_question, answer), trace) in enumerate(zip(records, traces)):
        question_type = trace.split(",")[-2]
        if question_type not in TYPES:
            continue
        context = " ".join(sentences).replace("_", " ")
        # ThoughtTracing's question cleaning and prompt (revised_tomi/test_tomi.py).
        question = raw_question.replace("_", " ").replace("-", " - ")
        match = next((m for pattern in QUESTION_PATTERNS if (m := pattern.fullmatch(question))), None)
        if match is None:
            raise ValueError(f"{source.stem}:{index}: unexpected question {question!r}")
        choices = tuple(dict.fromkeys(match[1] for match in _CONTAINER.finditer(context)))
        if answer not in choices:
            raise ValueError(f"{source.stem}:{index}: answer {answer!r} is not a container named in the story")
        order = question_type.split("_", 1)[0]
        examples.append(Example(
            id=f"{source.stem}:{index}", benchmark="paraphrased_tomi", context=context,
            question=question, choices=choices, answer=answer, target=match[1],
            category=question_type,
            belief_type="true_belief" if question_type.endswith("_no_tom") else "false_belief",
            answer_format="free_text", story_id=sha256(context.encode()).hexdigest(),
            native={"thoughttracing": {"dataset": "tomi",
                                       "input": f"{context.strip('.')}.\n\nQuestion: {question}\nAnswer:"},
                    "autotom": {"dataset_name": "ToMi-1st" if order == "first" else "ToMi-2nd"}},
        ))
    return examples


def load_examples(path):
    """Kept for older callers: every first- and second-order question in a file."""
    return load_all(path)


def load(source=SOURCE):
    return select_frozen("paraphrased_tomi", load_all(source), source)
