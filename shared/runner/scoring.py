"""Deterministic scoring, following each benchmark's released rule.

- AutoToM returns one of the options word for word, so it is correct when that is the answer.
- BigToM (ThoughtTracing's grade_bigtom): look for the option key "a)" / "b)" in the reply. The
  released grader checks the correct key first and, if neither key is found, asks GPT-4o. We
  instead accept a reply only when exactly one key (or exactly one option's text) is present,
  and mark it invalid otherwise, so the rule is symmetric and needs no grader model.
- MMToM-QA (ThoughtTracing's evaluate_answer): "(a)", "a)", "a.", "a:", "a," at the start, or
  "(a)" anywhere, or the bare letter. Same one-option-only rule.
- ParaphrasedToMi and ExploreToM (ThoughtTracing's ToMi rule): correct when the gold location
  appears in the reply, ignoring case. We also record whether another option was mentioned.
"""

import re


def _one(found):
    found = set(found)
    return found.pop() if len(found) == 1 else None


def _bigtom_choice(reply, choices):
    text = reply.strip().lower()
    if len(text) == 1:
        text += ")"
    keys = [letter for letter in "ab" if f"{letter})" in text]
    if keys:
        letter = _one(keys)
        return choices["ab".index(letter)] if letter else None
    return _one(choice for choice in choices if choice.lower().strip(".") in text)


def _mmtom_choice(reply, choices):
    text = reply.strip()

    def says(letter):
        return (any(text.startswith(letter + end) for end in ")" ".:,") or text.startswith(f"({letter})")
                or f"({letter})" in text or text == letter)
    letter = _one(letter for letter in "ab" if says(letter))
    if letter:
        return choices["ab".index(letter)]
    return _one(choice for choice in choices if choice.lower().strip(".") in text.lower())


def score(example, method, reply):
    """Return {"status", "correct", "choice", ...} for one reply."""
    if not isinstance(reply, str) or not reply.strip():
        return {"status": "invalid", "correct": False, "choice": None}
    if method == "autotom":
        valid = reply in example.choices
        return {"status": "ok" if valid else "invalid", "correct": valid and reply == example.answer,
                "choice": reply if valid else None}
    if example.benchmark in ("bigtom", "mmtom"):
        parse = _bigtom_choice if example.benchmark == "bigtom" else _mmtom_choice
        choice = parse(reply, example.choices)
        return {"status": "ok" if choice else "invalid", "correct": choice == example.answer,
                "choice": choice}
    text = reply.lower()
    others = [c for c in example.choices if c != example.answer and re.search(rf"\b{re.escape(c.lower())}\b", text)]
    return {"status": "ok", "correct": example.answer.lower() in text, "choice": None,
            "mentions_other_option": bool(others)}
