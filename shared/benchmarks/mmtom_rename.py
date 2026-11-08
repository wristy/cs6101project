"""Rename AutoToM's MMToM-QA rows to ThoughtTracing's anonymised names.

ThoughtTracing's revised MMToM-QA replaces object and room names with letters ("plate" becomes
"Object H", "kitchen" becomes "Room C") and says "fridge" for "refrigerator". AutoToM ships the
original names with pre-parsed states and actions. So that both methods see the same story, we
apply ThoughtTracing's mapping to AutoToM's story, question, choices, states and actions. The
mapping was read off the 600 aligned rows and is checked in tests/test_benchmarks.py.
"""

import re


OBJECTS = {
    "apple": "A", "bag of chips": "B", "chips": "B", "book": "C", "bottle of wine": "D", "wine": "D",
    "condiment bottle": "E", "cupcake": "F", "dish bowl": "G", "plate": "H", "remote control": "I",
    "salmon": "J", "water glass": "K", "wine glass": "L",
}
PLURALS = {"apples": "A", "bags of chips": "B", "books": "C", "bottles of wine": "D", "wines": "D",
           "condiment bottles": "E", "cupcakes": "F", "dish bowls": "G", "plates": "H",
           "remote controls": "I", "salmons": "J", "water glasses": "K", "wine glasses": "L"}
ROOMS = {"bathroom": "A", "bedroom": "B", "kitchen": "C", "living room": "D"}
ORDINALS = {"first": "1st", "second": "2nd", "third": "3rd", "fourth": "4th", "fifth": "5th",
            "sixth": "6th", "seventh": "7th", "eighth": "8th"}


def _alternation(words):
    return "|".join(re.escape(word) for word in sorted(words, key=len, reverse=True))


_PLURAL = re.compile(rf"\b({_alternation(PLURALS)})\b")
_OBJECT = re.compile(rf"\b(?:(an?|An?) )?({_alternation(OBJECTS)})\b")
_ROOM = re.compile(rf"\b({_alternation(ROOMS)})\b")


def rename(text):
    text = text.replace("refrigerator", "fridge")
    text = _PLURAL.sub(lambda m: f"Object {PLURALS[m[1]]}s", text)
    # Rooms first so that "kitchen cabinet" and "bathroom cabinet" keep their room letter.
    text = _ROOM.sub(lambda m: f"Room {ROOMS[m[1]]}", text)

    def object_name(match):
        article = match[1]
        name = f"Object {OBJECTS[match[2]]}"
        if article is None:
            return name
        return ("An " if article[0] == "A" else "an ") + name
    return _OBJECT.sub(object_name, text)


def rename_question_choice(text):
    """Choices in ThoughtTracing's questions number the cabinets ("1st Room C cabinet")."""
    text = rename(text)
    return re.sub(rf"\b({_alternation(ORDINALS)}) Room", lambda m: f"{ORDINALS[m[1]]} Room", text)
