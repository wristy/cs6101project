"""Experiment 5: replace near-duplicate hypotheses found by embeddings, instead of paraphrasing.

Upstream, after weighting an action: if the effective sample size is below n/2 it resamples;
otherwise, if word-overlap diversity is below 0.25, it paraphrases every hypothesis at
temperature 1. Here the resampling branch is unchanged, and the second branch becomes:

1. Embed the hypotheses with Qwen3-Embedding-8B and flag pairs with cosine similarity above
   THRESHOLD (0.931, calibrated on 30 labelled pairs from saved traces; see calibrate.py).
2. From each flagged pair keep the earlier hypothesis and drop the later one.
3. Generate replacements at temperature 0 that must differ from the kept hypotheses.
4. Re-score all four with the upstream weighting step and replace the weights.

The check runs at every action where upstream would have considered rejuvenation (no resample),
not only when Jaccard diversity is low. Embedding and generation calls count towards the arm's
cost. _trace() below is upstream's Tracer._trace word for word except that one branch
(tests/test_exp5_rejuvenation.py checks this).
"""

import os

from rich import print
import colorful as cf

from shared.bridge import extra_client
from shared.models.embeddings import EmbeddingClient, cosine

# Calibrated on 9 November (calibrate.py, 30 labelled pairs from the 3- and 10-question pilots):
# the lowest value that flagged no pair of genuinely different hypotheses. EXP5_THRESHOLD overrides it.
THRESHOLD = float(os.environ.get("EXP5_THRESHOLD", "0.931"))


def duplicate_slots(vectors, threshold=THRESHOLD):
    """Indices to replace: for each pair above the threshold, the later of the two (once)."""
    dropped = []
    for j in range(len(vectors)):
        for i in range(j):
            if i not in dropped and cosine(vectors[i], vectors[j]) > threshold:
                dropped.append(j)
                break
    return dropped


def similarity_matrix(vectors):
    return [[round(cosine(a, b), 4) for b in vectors] for a in vectors]


def build(upstream):
    compute_ess = upstream.compute_ess
    overall_jaccard_similarity = upstream.overall_jaccard_similarity
    resample_hypotheses_with_other_info = upstream.resample_hypotheses_with_other_info
    prompting_for_ordered_list = upstream.prompting_for_ordered_list

    class EmbeddingRejuvenationTracer(upstream.Tracer):
        def replace_duplicates(self, hypotheses, action):
            log = self.__dict__.setdefault("rejuvenation_log", [])
            embedder = extra_client("embed", lambda: EmbeddingClient(log_path=f"{self.args.output_dir}/calls-embed.jsonl"))
            vectors = embedder.embed(hypotheses.texts)
            dropped = duplicate_slots(vectors)
            entry = {"similarities": similarity_matrix(vectors), "threshold": THRESHOLD, "replaced": dropped}
            log.append(entry)
            if not dropped:
                return hypotheses
            kept = [text for index, text in enumerate(hypotheses.texts) if index not in dropped]
            history = self.interleave_context_and_perception(hypotheses.contexts, hypotheses.perceptions)
            existing = "\n".join(f"{n}. {text}" for n, text in enumerate(kept, 1))
            query = (f"{history}\n\n<existing hypotheses>\n{existing}\n</existing hypotheses>\n\n"
                     f"Generate a numbered list of {len(dropped)} new hypotheses on what were {self.target_agent}'s "
                     f"thoughts (e.g., beliefs, intent) that led to the latest action. Each must be clearly different "
                     f"from the existing hypotheses. Do not add any additional comments.")
            new = [text.strip() for text in prompting_for_ordered_list(self.tracer_model, prompt=query, n=len(dropped))]
            if len(new) < len(dropped):
                entry["error"] = "too few replacements"
                return hypotheses
            texts = list(hypotheses.texts)
            for index, text in zip(dropped, new):
                texts[index] = text
            hypotheses.update_texts(texts)
            weight_results = self.weigh(hypotheses, action, mode="prompting")
            hypotheses.update_weights(weight_results["weights"])
            hypotheses.weight_details = weight_results
            entry.update(new_texts=new, new_weights=list(map(float, weight_results["weights"])))
            return hypotheses

        def chain_weighted_average_trace(self, hypotheses_list):
            result = super().chain_weighted_average_trace(hypotheses_list)
            result["rejuvenation_log"] = self.__dict__.get("rejuvenation_log", [])
            return result

        def _trace(self, text: str, target_agent=None):
            preprocessed_text = self.preprocess_input(text, target_agent)
            if preprocessed_text is None:
                print(cf.bold | cf.magenta("Failed to identify the target agent."))
                self.dump({'summary': ""}, [])
                return ""
            self.set_tracer_variables(preprocessed_text)

            trajectory = preprocessed_text['trajectory']
            perceptions_trajectory = preprocessed_text['perceptions']

            hypotheses_list = []
            context_history = []
            for idx, (state_action, perceptions) in enumerate(zip(trajectory, perceptions_trajectory)):
                if idx == 0:
                    new_hypotheses = self.initialize(state_action=state_action, perceptions=perceptions)
                else:
                    existing_hypotheses = hypotheses_list[-1]
                    new_hypotheses = self.propagate(existing_hypotheses, state_action=state_action, perceptions=perceptions)

                if state_action['action']:
                    weight_results = self.weigh(new_hypotheses, state_action['action'], mode="prompting")
                    new_hypotheses.update_weights(weight_results['weights'])
                    new_hypotheses.weight_details = weight_results

                    # resample hypotheses or jitter
                    if self.args.n_hypotheses > 1:
                        ess = compute_ess(new_hypotheses)
                        overall_text_diversity = 1 - overall_jaccard_similarity(new_hypotheses.texts)
                        if ess < (self.args.n_hypotheses) / 2:
                            new_hypotheses = resample_hypotheses_with_other_info(new_hypotheses, ess)
                        else:
                            new_hypotheses = self.replace_duplicates(new_hypotheses, state_action['action'])
                else:
                    pass

                hypotheses_list.append(new_hypotheses)

                # update history
                if state_action['state']:
                    context_history.append({'text': state_action['state'], 'action': False})
                if state_action['action']:
                    context_history.append({'text': state_action['action'], 'action': True})

            traced_thoughts = self.chain_weighted_average_trace(hypotheses_list)
            trace_text = f"{self.trace_header}\n\n{traced_thoughts['text']}"

            self.dump(traced_thoughts, hypotheses_list)
            return trace_text

    return EmbeddingRejuvenationTracer
