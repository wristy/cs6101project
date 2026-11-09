"""Experiment 2a: Jev picks a mental model from the story; hypotheses fill that model's slots.

Before tracing, Jev reads the story only (never the question or the assumption sentence) and
returns a probability for each of four AutoToM-style mental models, asked with Jev's choice
question (unordered named options, each with a description). We take the most probable model and
keep the whole distribution in the trace. The model fixes the slots every hypothesis fills, and
plain code joins the slots into one sentence for the rest of ThoughtTracing.
"""

from shared.bridge import extra_client
from shared.tracing.structured import build_structured


def _slots(*names):
    return {"type": "object", "additionalProperties": False, "required": list(names),
            "properties": {name: {"type": "string"} for name in names}}


class MentalModel:
    def __init__(self, name, description, slots, template, guide):
        self.name, self.description, self.schema = name, description, _slots(*slots)
        self.template, self.guide = template, guide

    def instructions(self, agent):
        return (f"Write each hypothesis by filling these slots about {agent}: {self.guide.format(agent=agent)} "
                f"A hypothesis is a JSON object with the keys {', '.join(self.schema['required'])}; "
                "each value is a short phrase.")

    def render(self, agent, slots):
        return self.template.format(agent=agent, **{k: v.strip().rstrip(".") for k, v in slots.items()})


MODELS = [
    MentalModel("belief", "Belief only: the person's actions follow from what they perceived and therefore believe; "
                "what they want is obvious or plays no part.",
                ["observation", "belief"], "{agent} observed {observation}, so {agent} believes {belief}.",
                "observation (what {agent} perceived), belief (what {agent} therefore believes)."),
    MentalModel("goal", "Goal only: the person's actions follow from what they want; what they believe about the "
                "world is obvious or plays no part.",
                ["goal", "plan"], "{agent} wants {goal}, so {agent} plans to {plan}.",
                "goal (what {agent} wants), plan (what {agent} does or will do because of it)."),
    MentalModel("belief_goal", "Belief and goal: the person acts on what they want and on what they believe about where or "
                "how things are, and those beliefs may be incomplete or wrong.",
                ["observation", "belief", "goal"],
                "{agent} observed {observation}, so {agent} believes {belief}; {agent} wants {goal}.",
                "observation (what {agent} perceived), belief (what {agent} therefore believes), "
                "goal (what {agent} wants)."),
    MentalModel("higher_order", "Higher-order: the person reasons about what another person believes or wants.",
                ["other_person", "observation", "belief_about_other"],
                "{agent} observed {observation}, so {agent} believes that {other_person} {belief_about_other}.",
                "other_person (whose mind {agent} is thinking about), observation (what {agent} perceived), "
                "belief_about_other (what {agent} believes that person thinks or wants)."),
]
# Reworded after the pilot: "pick the simplest kind that is enough" pushed every MMToM-QA story to
# goal-only, although its stories mix belief and goal questions about a person searching a home
# with partial knowledge. The question now asks what explains the person's actions.
JEV_INSTRUCTIONS = ("Which mental states must be tracked to explain and predict the main person's actions in "
                    "this story: what they believe, what they want, both, or what they think others "
                    "believe or want?")


def choose_model(jev, story):
    answer = jev.choice(story, JEV_INSTRUCTIONS, {model.name: model.description for model in MODELS})
    probabilities = answer["probabilities"]
    # Take the argmax ourselves (ties go to the simpler model) and keep Jev's own pick for the record.
    best = max(MODELS, key=lambda model: probabilities[model.name])
    return best, {"probabilities": probabilities, "jev_choice": answer["choice"],
                  "confidence": answer.get("confidence"), "chosen": best.name}


def build(upstream):
    from shared.models.typesafe import JevClient

    class MentalModelTracer(build_structured(upstream)):
        def set_tracer_variables(self, preprocessed_text):
            super().set_tracer_variables(preprocessed_text)
            # The story only: preprocess_input already cut the question off ("context").
            story = preprocessed_text["context"]
            if self.args.dataset == "mmtom":
                story = story.split("\nQuestion:")[0]
            jev = extra_client("jev", lambda: JevClient(log_path=f"{self.args.output_dir}/calls-jev.jsonl"))
            self.mental_model, self.mental_model_choice = choose_model(jev, story.strip())

        def hypothesis_format(self):
            return self.mental_model

        def chain_weighted_average_trace(self, hypotheses_list):
            result = super().chain_weighted_average_trace(hypotheses_list)
            result["mental_model"] = self.mental_model_choice
            return result

    return MentalModelTracer
