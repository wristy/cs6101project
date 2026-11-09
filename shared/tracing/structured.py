"""Structured hypotheses for Experiments 2a and 2b.

ThoughtTracing's hypotheses are free text. Here the model fills a fixed structure (slots for 2a,
relation triples for 2b) as JSON, and plain code joins it into one sentence. That sentence is
the hypothesis the rest of ThoughtTracing sees, so weighting, resampling, rejuvenation and the
final answer are unchanged. Only two steps change:

- initialize(): same input as upstream, but asks for the n hypotheses in the structure.
- propagate(): same input as upstream, but asks for the updated hypothesis in the structure,
  showing the previous one in its structured form when we have it.

`response_format` (JSON schema) is off on the first try: as agreed, the schema is enforced only
once the model breaks the structure, so retries (up to three tries in all, as upstream's
prompting_for_ordered_list) send it. MiMo passed the 10-story format check without it, but broke
the 2b structure on about a quarter of the full run's questions. A list longer than its
maxItems (more than n hypotheses, or more than 4 triples) keeps its first items, as upstream
keeps the first n hypotheses. Every try is logged in structure_log.
"""

from copy import deepcopy
import json

import numpy as np

from shared.bridge import get_client


class StructureError(ValueError):
    pass


def validate(value, schema, defs=None):
    """A small JSON-schema check for the subset our schemas use."""
    defs = defs if defs is not None else schema.get("$defs", {})
    if "$ref" in schema:
        return validate(value, defs[schema["$ref"].split("/")[-1]], defs)
    if "anyOf" in schema:
        for option in schema["anyOf"]:
            try:
                return validate(value, option, defs)
            except StructureError:
                continue
        raise StructureError(f"{value!r} matches no allowed form")
    kind = schema.get("type")
    if kind == "string":
        if not isinstance(value, str) or not value.strip():
            raise StructureError(f"expected a non-empty string, got {value!r}")
        if "enum" in schema and value not in schema["enum"]:
            raise StructureError(f"{value!r} is not one of {schema['enum']}")
    elif kind == "array":
        if not isinstance(value, list):
            raise StructureError(f"expected a list, got {value!r}")
        if not schema.get("minItems", 0) <= len(value) <= schema.get("maxItems", len(value)):
            raise StructureError(f"list has {len(value)} items")
        for item in value:
            validate(item, schema["items"], defs)
    elif kind == "object":
        if not isinstance(value, dict):
            raise StructureError(f"expected an object, got {value!r}")
        missing = set(schema.get("required", [])) - set(value)
        extra = set(value) - set(schema.get("properties", {}))
        if missing or (extra and schema.get("additionalProperties") is False):
            raise StructureError(f"missing {sorted(missing)} or unexpected {sorted(extra)}")
        for key, item in value.items():
            if key in schema.get("properties", {}):
                validate(item, schema["properties"][key], defs)
    return value


def parse_json(text):
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0]
    try:
        return json.loads(text)
    except json.JSONDecodeError as error:
        raise StructureError(f"not JSON: {error}") from error


def trim(value, schema, defs=None):
    """Keep the first maxItems of any list that is too long, as upstream's
    prompting_for_ordered_list keeps the first n hypotheses when given more. Nothing else is
    changed; validate() still rejects everything else that breaks the schema."""
    defs = defs if defs is not None else schema.get("$defs", {})
    if "$ref" in schema:
        return trim(value, defs[schema["$ref"].split("/")[-1]], defs)
    if schema.get("type") == "array" and isinstance(value, list):
        if "maxItems" in schema:
            value = value[:schema["maxItems"]]
        return [trim(item, schema.get("items", {}), defs) for item in value]
    if schema.get("type") == "object" and isinstance(value, dict):
        properties = schema.get("properties", {})
        return {key: trim(item, properties[key], defs) if key in properties else item for key, item in value.items()}
    return value


def list_schema(item_schema, n):
    """{"hypotheses": [n items]}, keeping item $defs at the top level."""
    item = {key: value for key, value in item_schema.items() if key != "$defs"}
    schema = {"type": "object", "additionalProperties": False, "required": ["hypotheses"],
              "properties": {"hypotheses": {"type": "array", "minItems": n, "maxItems": n, "items": item}}}
    if "$defs" in item_schema:
        schema["$defs"] = item_schema["$defs"]
    return schema


def build_structured(upstream):
    """A Tracer subclass whose hypotheses follow self.hypothesis_format()."""
    HypothesesSetV3 = upstream.HypothesesSetV3

    class StructuredTracer(upstream.Tracer):
        use_response_format = False
        tolerance = 3

        def hypothesis_format(self):
            """Return an object with .name, .schema, .instructions(agent), .render(agent, value)."""
            raise NotImplementedError

        def ask_structured(self, prompt, schema, system_prompt=None):
            log = self.__dict__.setdefault("structure_log", [])
            enforced = {"type": "json_schema", "json_schema": {"name": "hypotheses", "schema": schema, "strict": True}}
            temperature, error = 0, None
            for attempt in range(self.tolerance):
                # As agreed: ask plainly first; enforce the schema only once the model has broken it.
                response_format = enforced if (self.use_response_format or attempt > 0) else None
                reply = get_client().generate(prompt, system_prompt=system_prompt, temperature=temperature,
                                              max_tokens=1024, response_format=response_format)
                try:
                    value = validate(trim(parse_json(reply), schema), schema)
                    log.append({"ok": True, "attempt": attempt, "enforced": response_format is not None})
                    return value
                except StructureError as failure:
                    error = failure
                    log.append({"ok": False, "attempt": attempt, "error": str(failure), "reply": reply[:2000]})
                    temperature += 0.33   # as upstream prompting_for_ordered_list
            raise StructureError(f"no valid structured reply after {self.tolerance} tries: {error}")

        def _remember(self, value, text):
            self.__dict__.setdefault("structured", {})[text] = value
            return text

        def initialize(self, state_action, perceptions):
            # Upstream Tracer.initialize builds this input; the request asks for structure instead.
            context_input = ""
            state, action = state_action['state'], state_action['action']
            if state:
                agent_state = self.get_agent_state(self.target_agent, state)
                context_input += f"<state>\n{agent_state.strip()}\n</state>\n"
                context_input += f"<note>{perceptions['state']}</note>\n\n"
            if action:
                context_input += f"<action>\n{action}\n</action>\n"
                if perceptions['action']:
                    context_input += f"<note>{perceptions['action']}</note>\n"
            n = self.args.n_hypotheses
            fmt = self.hypothesis_format()
            if action:
                ask = f"Generate {n} hypotheses on what were {self.target_agent}'s thoughts (e.g., beliefs, intent) that led to the action above."
            else:
                ask = f"Generate {n} hypotheses on what {self.target_agent} will be thinking (e.g., beliefs)."
            query = (f"{context_input.strip()}{self.assumption}\n\n{ask} {fmt.instructions(self.target_agent)}\n"
                     f"Answer with JSON only: {{\"hypotheses\": [{n} hypotheses]}}.")
            values = self.ask_structured(query, list_schema(fmt.schema, n))["hypotheses"]
            texts = [self._remember(value, fmt.render(self.target_agent, value)) for value in values]
            weights = np.ones(len(texts)) / len(texts)
            return HypothesesSetV3(target_agent=self.target_agent, contexts=[state_action],
                                   perceptions=[perceptions], texts=texts, weights=weights)

        def propagate(self, existing_hypotheses, state_action, perceptions):
            prop_info = self.setup_propagation(existing_hypotheses, state_action, perceptions)
            target_agent = prop_info["target_agent"]
            fmt = self.hypothesis_format()
            system_prompt = f"You are an expert assistant trying to predict {target_agent}'s thoughts."
            known = self.__dict__.get("structured", {})
            texts = []
            for hypothesis in existing_hypotheses.texts:
                previous = json.dumps(known[hypothesis]) if hypothesis in known else hypothesis
                prompt = (f"{self.trace_header}\n\n<previous context>\n{prop_info['context_and_perception_str']}\n"
                          f"</previous context>\n<previous prediction regarding {target_agent}'s thoughts>\n{previous}\n"
                          f"</previous prediction regarding {target_agent}'s thoughts>\n\n<current context>"
                          f"{self.assumption}\n{prop_info['new_context']}\n</current context>\n\n"
                          f"Question: What did {target_agent} believe? Update the prediction. "
                          f"{fmt.instructions(target_agent)}\nAnswer with JSON only: one hypothesis.")
                value = self.ask_structured(prompt, fmt.schema, system_prompt=system_prompt)
                texts.append(self._remember(value, fmt.render(target_agent, value)))
            return HypothesesSetV3(target_agent, prop_info["context_history"], prop_info["perception_history"],
                                   texts, existing_hypotheses.weights, parent_hypotheses=existing_hypotheses.hypotheses)

        def chain_weighted_average_trace(self, hypotheses_list):
            result = super().chain_weighted_average_trace(hypotheses_list)
            result["structure_log"] = self.__dict__.get("structure_log", [])
            result["structured"] = self.__dict__.get("structured", {})
            return result

    return StructuredTracer
