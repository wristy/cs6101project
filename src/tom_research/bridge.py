"""Process-local connection used by the small upstream compatibility patches."""

import os


_client = None


def configure(client):
    global _client
    _client = client
    os.environ["TOM_RESEARCH_BACKEND"] = "1"


def get_client():
    if _client is None:
        raise RuntimeError("Research model client has not been configured")
    return _client


class SharedAgent:
    def __init__(self, args):
        self.args = args

    def interact(self, prompt, temperature=0, max_tokens=None, system_prompt=None, history=None):
        return get_client().generate(prompt, temperature=temperature or 0, max_tokens=max_tokens,
                                     system_prompt=system_prompt, history=history)

    def batch_interact(self, prompts, temperature=0, max_tokens=None, system_prompts=None, histories=None):
        if system_prompts is None or isinstance(system_prompts, str):
            system_prompts = [system_prompts] * len(prompts)
        histories = histories if histories is not None else [None] * len(prompts)
        if len(system_prompts) != len(prompts) or len(histories) != len(prompts):
            raise ValueError("Batch lengths differ")
        return [self.interact(prompt, temperature, max_tokens, system_prompt, history)
                for prompt, system_prompt, history in zip(prompts, system_prompts, histories)]