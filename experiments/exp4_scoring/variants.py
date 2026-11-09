"""Experiment 4: two new ways to weight ThoughtTracing's hypotheses by the agent's next action.

Everything else in ThoughtTracing is unchanged, including that weights are replaced (not
multiplied) at each action and the resample/rejuvenate rule that follows. Only
prompt_likelihood() is replaced; it builds the same context string as upstream, then:

- AutoToM-style ("ab"): AutoToM's P(Action | Goal, Belief) prompt from probs.py, word for word,
  with the hypothesis as the belief, answered with one letter and scored by the agreed A/B rule
  (P(A), or 1 - P(B) if the model writes B; AutoToM's 0.1 if it writes neither).
- Jev ("noul"): TypeSafe's probability that the statement "Given the context and {agent}'s
  thoughts [...], {agent} would do the next action: ..." is true, with the context as Jev's state.
  No text is generated.

Both normalise the four probabilities to sum to 1 (uniform if all are 0) and record the raw
probabilities and the weight spread (max - min) for the analysis.
"""

import numpy as np

from shared.bridge import extra_client, get_client

# AutoToM probs.py, get_likelihood_general, "Action" without goal-of-others (P(Action | Goal, Belief)).
AUTOTOM_ACTION = """Determine if the statement is likely, respond with only either A or B.
{info}
Here is a statement of {inf_agent}'s action. The belief stands for {inf_agent}'s current belief which is true. {inf_agent} is likely to act according to goal and belief concerning certain objects (the wording for objects must be same. You should ignore the correlation of different objects. e.g., plate and apple are two different objects.) Notice that {inf_agent}'s belief does not represent the goal.
When belief and goal are irrelevant, and action is directly driven by goal, it's likely. When belief and goal are relevant (about exactly the same object) and they contradict (or irrelevant) with action, it's unlikely.
Determine if the following statement is likely: {statement}
A) Likely.
B) Unlikely."""
NOUL_STATEMENT = ("Given the context and {agent}'s thoughts [{hypothesis}], "
                  "{agent} would do the next action: {action}")


def normalise(probabilities):
    probabilities = np.asarray(probabilities, dtype=float)
    total = probabilities.sum()
    if total <= 0:
        return np.full(len(probabilities), 1 / len(probabilities))
    return probabilities / total


def _context(tracer, context_history, perception_history):
    """Upstream prompt_likelihood's context: history without the action being scored."""
    from copy import deepcopy
    contexts, perceptions = deepcopy(context_history), deepcopy(perception_history)
    contexts[-1]["action"] = None
    perceptions[-1]["action"] = None
    if len(contexts) > 1:
        return tracer.interleave_context_and_perception(contexts, perceptions)
    return contexts[-1]["state"] or ""


def _results(prompts, probabilities):
    weights = normalise(probabilities)
    # Upstream's resample step copies weight_details["reasonings"] per resampled hypothesis; these
    # scorers generate no reasoning text, so the raw probability stands in for it.
    return {"prompts": prompts, "reasonings": [f"P = {p:.4f}" for p in probabilities],
            "raw_scores": np.asarray(probabilities, dtype=float),
            "weights": weights, "weight_spread": float(weights.max() - weights.min())}


def build_ab(upstream):
    class AutoToMScoredTracer(upstream.Tracer):
        def prompt_likelihood(self, existing_hypotheses, context_history, perception_history, action, target_agent=None):
            agent = self.target_agent if target_agent is None else target_agent
            context = _context(self, context_history, perception_history)
            prompts = [AUTOTOM_ACTION.format(info=f"State: {context}\n{agent}'s Belief: {hypothesis}",
                                             inf_agent=agent, statement=action)
                       for hypothesis in existing_hypotheses]
            client = get_client()
            return _results(prompts, [client.score_label(prompt, fallback=0.1) for prompt in prompts])
    return AutoToMScoredTracer


def build_noul(upstream):
    from shared.models.typesafe import JevClient

    class JevScoredTracer(upstream.Tracer):
        def prompt_likelihood(self, existing_hypotheses, context_history, perception_history, action, target_agent=None):
            agent = self.target_agent if target_agent is None else target_agent
            context = _context(self, context_history, perception_history)
            jev = extra_client("jev", lambda: JevClient(log_path=f"{self.args.output_dir}/calls-jev.jsonl"))
            statements = [NOUL_STATEMENT.format(agent=agent, hypothesis=hypothesis, action=action)
                          for hypothesis in existing_hypotheses]
            return _results(statements, [jev.noul(context, statement) for statement in statements])
    return JevScoredTracer
