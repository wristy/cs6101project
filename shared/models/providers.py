"""The project's models, by short name. Use model_config("mimo") or model_config("gpt6sol").

Settings agreed on 8 November:
- MiMo-V2.6-Pro on DeepInfra: thinking off via the chat template flag, temperature 0.
- GPT-6 Sol on OpenAI's own API: reasoning effort "none" (which allows temperature 0 and logprobs);
  it needs max_completion_tokens and at least ~16 output tokens even for a one-letter answer.
- Qwen3-Embedding-8B on DeepInfra for Experiment 5.
Jev and Noul (TypeSafe, one key: JEV_API_KEY) are added with Experiments 2a and 4.
"""

from dataclasses import replace

from shared.models.client import ModelConfig

CHAT_MODELS = {
    "mimo": ModelConfig(),
    "gpt6sol": ModelConfig(
        model="gpt-6-sol", provider="openai", base_url="https://api.openai.com/v1", key_env="OPENAI_API_KEY",
        token_param="max_completion_tokens", scoring_max_tokens=16, extra={"reasoning_effort": "none"},
    ),
}

EMBEDDING_MODEL = {"model": "Qwen/Qwen3-Embedding-8B", "base_url": "https://api.deepinfra.com/v1/openai",
                   "key_env": "DEEPINFRA_API_KEY"}


def model_config(name, **overrides):
    if name not in CHAT_MODELS:
        raise KeyError(f"Unknown model {name!r}; choose from {sorted(CHAT_MODELS)}")
    return replace(CHAT_MODELS[name], **overrides)
