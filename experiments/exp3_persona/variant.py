"""Experiment 3: the propagation step's system prompt casts the model as a neutral observer.

propagate() below is ThoughtTracing's Tracer.propagate word for word, except for the system
prompt. tests/test_exp3_persona.py checks that against the pinned upstream source.
"""

ORIGINAL = "You are an expert assistant trying to predict {target_agent}'s thoughts."
OBSERVER = "You are a neutral observer watching {target_agent} and trying to predict {target_agent}'s thoughts."


def build(upstream):
    HypothesesSetV3 = upstream.HypothesesSetV3

    class ObserverTracer(upstream.Tracer):
        def propagate(self, existing_hypotheses, state_action, perceptions):
            prop_info = self.setup_propagation(existing_hypotheses, state_action, perceptions)
            target_agent = prop_info["target_agent"]
            context_history = prop_info["context_history"]
            perception_history = prop_info["perception_history"]
            context_and_perception_str = prop_info["context_and_perception_str"]
            new_context = prop_info["new_context"]

            system_prompt = OBSERVER.format(target_agent=target_agent)
            propagation_prompts = [f"{self.trace_header}\n\n<previous context>\n{context_and_perception_str}\n</previous context>\n<previous prediction regarding {target_agent}'s thoughts>\n{hypothesis}\n</previous prediction regarding {target_agent}'s thoughts>\n\n<current context>{self.assumption}\n{new_context}\n</current context>\n\nQuestion: What did {target_agent} believe?" for hypothesis in existing_hypotheses.texts]
            propagated_texts = self.tracer_model.batch_interact(propagation_prompts, system_prompts=system_prompt, temperature=0, max_tokens=1024)
            propagated_hypotheses = HypothesesSetV3(target_agent, context_history, perception_history, propagated_texts, existing_hypotheses.weights, parent_hypotheses=existing_hypotheses.hypotheses)

            return propagated_hypotheses

    return ObserverTracer
