"""Experiment 2b: each hypothesis is a small set of (subject, relation, object) triples.

Relations are the report's eight (Sees, Knows, Believes, IsUnawareOf, Wants, Likes, Dislikes,
LocatedIn) plus Moved. LocatedIn may only appear inside Knows or Believes. An object can be
another triple, up to three levels deep, which expresses higher-order beliefs:
(Anne, Believes, (Sally, Believes, (apple, LocatedIn, basket))). Code turns the triples into one
sentence, which is what the rest of ThoughtTracing sees.
"""

from shared.tracing.structured import build_structured

RELATIONS = ["Sees", "Knows", "Believes", "IsUnawareOf", "Wants", "Likes", "Dislikes", "Moved"]
NESTED_ONLY = ["LocatedIn"]
PHRASES = {"Sees": "sees", "Knows": "knows that", "Believes": "believes that", "IsUnawareOf": "is unaware of",
           "Wants": "wants", "Likes": "likes", "Dislikes": "dislikes", "Moved": "moved",
           "LocatedIn": "is in"}
MAX_TRIPLES = 4


def _triple(name, relations, inner):
    obj = {"type": "string"} if inner is None else {"anyOf": [{"type": "string"}, {"$ref": f"#/$defs/{inner}"}]}
    return {"type": "object", "additionalProperties": False, "required": ["subject", "relation", "object"],
            "properties": {"subject": {"type": "string"}, "relation": {"type": "string", "enum": relations},
                           "object": obj}}


SCHEMA = {
    "type": "object", "additionalProperties": False, "required": ["triples"],
    "properties": {"triples": {"type": "array", "minItems": 1, "maxItems": MAX_TRIPLES,
                               "items": {"$ref": "#/$defs/top"}}},
    "$defs": {"top": _triple("top", RELATIONS, "level2"),
              "level2": _triple("level2", RELATIONS + NESTED_ONLY, "level3"),
              "level3": _triple("level3", RELATIONS + NESTED_ONLY, None)},
}


def render_triple(triple):
    obj = triple["object"]
    obj_text = render_triple(obj) if isinstance(obj, dict) else obj
    return f"{triple['subject']} {PHRASES[triple['relation']]} {obj_text}"


def render(agent, hypothesis):
    sentences = [render_triple(triple) for triple in hypothesis["triples"]]
    return ". ".join(sentence[0].upper() + sentence[1:] for sentence in sentences) + "."


class Triples:
    name = "triples"
    schema = SCHEMA

    @staticmethod
    def instructions(agent):
        return (f"Write each hypothesis as 1 to {MAX_TRIPLES} (subject, relation, object) triples about "
                f"{agent} and the people, objects and containers in the story. The relation is one of "
                f"{', '.join(RELATIONS)}; LocatedIn may be used only inside a Knows or Believes triple. "
                "An object can itself be a triple, e.g. (Anne, Believes, (Sally, Believes, (apple, LocatedIn, "
                "basket))). A hypothesis is {\"triples\": [...]}, each triple {\"subject\": ..., \"relation\": ..., "
                "\"object\": text or triple}.")

    render = staticmethod(render)


def build(upstream):
    class TripleTracer(build_structured(upstream)):
        def hypothesis_format(self):
            return Triples
    return TripleTracer
