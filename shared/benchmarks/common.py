"""The question record every benchmark loader returns, and the frozen ID lists.

A frozen list (data/<benchmark>.json) names the exact questions we run, the source file they
come from and that file's sha256, so every experiment uses the same questions and a changed
source file is caught instead of silently giving a different set.
"""

from dataclasses import asdict, dataclass, field
from hashlib import sha256
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
EXTERNAL = ROOT / "external"
FROZEN = Path(__file__).resolve().parent / "data"


@dataclass(frozen=True)
class Example:
    id: str
    benchmark: str
    context: str
    question: str                      # the bare question, without answer options
    choices: tuple[str, ...]           # answer options as plain text
    answer: str                        # the correct option, word for word
    target: str                        # the agent whose mind the question is about
    category: str                      # the benchmark's own question type
    belief_type: str = ""              # "false_belief", "true_belief" or "" when not applicable
    answer_format: str = "multiple_choice"
    story_id: str = ""
    native: dict = field(default_factory=dict, compare=False)  # per-method inputs, never the answer

    def inference_input(self):
        """What a method may see: everything except the gold answer."""
        return {key: value for key, value in asdict(self).items()
                if key not in {"answer", "category", "belief_type", "story_id"}}


def file_sha256(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def freeze(benchmark, examples, source, note=""):
    """Write the frozen ID list for a benchmark."""
    FROZEN.mkdir(parents=True, exist_ok=True)
    record = {"benchmark": benchmark, "source": str(Path(source).resolve().relative_to(ROOT)),
              "source_sha256": file_sha256(source), "note": note, "count": len(examples),
              "ids": [example.id for example in examples]}
    path = FROZEN / f"{benchmark}.json"
    path.write_text(json.dumps(record, indent=1) + "\n")
    return path


def select_frozen(benchmark, examples, source):
    """Return the frozen questions, in frozen order, after checking the source file."""
    record = json.loads((FROZEN / f"{benchmark}.json").read_text())
    if file_sha256(source) != record["source_sha256"]:
        raise ValueError(f"{benchmark}: source file {source} differs from the frozen one")
    by_id = {example.id: example for example in examples}
    missing = [question_id for question_id in record["ids"] if question_id not in by_id]
    if missing:
        raise ValueError(f"{benchmark}: frozen IDs missing from source: {missing[:5]}")
    return [by_id[question_id] for question_id in record["ids"]]
