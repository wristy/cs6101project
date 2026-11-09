# API probes

These are two one-off checks we ran before the experiments. They decided how AutoToM-style "A) Likely / B) Unlikely"
probabilities are read from each API. Their saved outputs sit next to the scripts.

- **`logprob_probe.py` → `probe_results.json`.** It checks whether an API returns the probabilities of
  tokens other than the one the model wrote, and whether `logit_bias` works. MiMo on DeepInfra returns
  only the chosen token's probability. So we use P(A) when the model writes A and 1 − P(B) when it
  writes B (`shared/models/`).
- **`autotom_likelihood_probe.py` → `autotom_likelihood_results.json`.** It sends 20 of AutoToM's own
  likelihood prompts to each model. MiMo's probabilities are graded, with a median of 0.96 and 8 of 20
  below 0.9. GPT-6 Sol's are nearly always about 1.0, so for GPT-6 Sol AutoToM's weights are close to
  hard votes.

Keys are read from the environment, for example `DEEPINFRA_API_KEY=... python logprob_probe.py`.
