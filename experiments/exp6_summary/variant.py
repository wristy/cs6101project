"""Experiment 6: propagation and weighting read a running perception summary, not the raw history.

Upstream, every propagation and weighting call reads the whole story so far, interleaved with
perception notes. Here one shared summary per question is updated after each step by the LLM:
what the agent has perceived so far and what that implies about the world. It is built only
from the steps themselves (the same perception-filtered rendering upstream uses), so it never
sees the question or the assumption sentence.

- Propagation reads the summary of all earlier steps (the new step is shown as upstream).
- Weighting reads the summary of earlier steps plus the current step rendered as upstream.
- The final answer still reads the full trace: weighted_average_hypotheses is unchanged.
"""


from shared.bridge import get_client


def build(upstream):
    class SummaryTracer(upstream.Tracer):
        def _raw_step(self, context, perception):
            return upstream.Tracer.interleave_context_and_perception(self, [context], [perception])

        def summary(self, contexts, perceptions):
            """Running summary after len(contexts) steps, computed once per step and cached."""
            cache = self.__dict__.setdefault("summaries", {(): ""})
            steps = []
            summary = ""
            for context, perception in zip(contexts, perceptions):
                if not (context["state"] or context["action"]):
                    continue
                step = self._raw_step(context, perception)
                steps.append(step)
                key = tuple(steps)
                if key not in cache:
                    agent = self.target_agent
                    cache[key] = get_client().generate(
                        f"<previous summary>\n{summary}\n</previous summary>\n\n<new step>\n{step}\n</new step>\n\n"
                        f"Update the summary of what {agent} has perceived so far and what that implies about the "
                        f"world from {agent}'s point of view. Keep only what matters for {agent}'s beliefs and "
                        f"goals, such as what {agent} has seen and where {agent} has already looked. Keep it short. "
                        "Do not add any additional comments.",
                        system_prompt=f"You keep a short running summary of what {agent} has perceived.",
                        temperature=0, max_tokens=512)
                    self.__dict__.setdefault("summary_log", []).append({"steps": len(steps), "summary": cache[key]})
                summary = cache[key]
            return summary

        def _summary_block(self, contexts, perceptions):
            text = self.summary(contexts, perceptions).strip()
            agent = self.target_agent
            return f"<summary of what {agent} has perceived so far>\n{text}\n</summary of what {agent} has perceived so far>" if text else ""

        def setup_propagation(self, existing_hypotheses, state_action, perceptions):
            info = super().setup_propagation(existing_hypotheses, state_action, perceptions)
            info["context_and_perception_str"] = self._summary_block(existing_hypotheses.contexts,
                                                                     existing_hypotheses.perceptions)
            return info

        def weigh(self, hypotheses, action, mode):
            self._weighing = True
            try:
                return super().weigh(hypotheses, action, mode)
            finally:
                self._weighing = False

        def interleave_context_and_perception(self, context_history, perception_history, target_agent=None, chat=False):
            if not getattr(self, "_weighing", False):
                return super().interleave_context_and_perception(context_history, perception_history, target_agent, chat)
            earlier = self._summary_block(context_history[:-1], perception_history[:-1])
            current = super().interleave_context_and_perception(context_history[-1:], perception_history[-1:],
                                                                target_agent, chat)
            return f"{earlier}\n\n{current}".strip()

        def chain_weighted_average_trace(self, hypotheses_list):
            result = super().chain_weighted_average_trace(hypotheses_list)
            result["summary_log"] = self.__dict__.get("summary_log", [])
            return result

    return SummaryTracer
