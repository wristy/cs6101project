"""One logged, budgeted client for every chat model in the project.

Providers are configured in shared/models/providers.py. Two kinds of call:

- generate(): text, with thinking off and a check that the reply finished cleanly.
- score_label(): the agreed A/B probability rule for AutoToM-style likelihoods. The model answers
  one letter; we read the logprob of the token it wrote and return P(A) if it wrote "A" and
  1 - P(B) if it wrote "B". This is the rule from the progress report. Probes showed MiMo on
  DeepInfra never returns alternative tokens and ignores logit_bias, and GPT-6 Sol rejects
  logit_bias; when GPT-6 Sol does return both letters they sum to ~1.0, so the rule matches
  reading both. See probes/.
"""

from dataclasses import asdict, dataclass, field
import json
import math
import os
from pathlib import Path
import time

from openai import APIConnectionError, APITimeoutError, InternalServerError, OpenAI, RateLimitError


class ProtocolError(RuntimeError):
    pass


@dataclass(frozen=True)
class ModelConfig:
    """Everything needed to call one model; stored with every result so runs are reproducible."""
    model: str = "XiaomiMiMo/MiMo-V2.6-Pro"
    provider: str = "deepinfra"
    base_url: str = "https://api.deepinfra.com/v1/openai"
    key_env: str = "DEEPINFRA_API_KEY"
    token_param: str = "max_tokens"          # OpenAI's GPT-6 models take max_completion_tokens
    scoring_max_tokens: int = 1              # GPT-6 Sol needs ~16; only the first token is read
    extra: dict = field(default_factory=lambda: {"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}})
    seed: int = 42
    max_tokens: int = 4096
    timeout: float = 180
    max_retries: int = 2
    max_calls: int = 500
    top_logprobs: int = 5                    # OpenAI caps this at 5


TRANSIENT = (APIConnectionError, APITimeoutError, InternalServerError, RateLimitError)


class LLMClient:
    def __init__(self, config, transport=None, log_path=None):
        self.config = config
        self.records = []
        self.log_path = Path(log_path) if log_path else None
        if transport is None:
            key = os.getenv(config.key_env)
            if not key:
                raise RuntimeError(f"{config.key_env} is not set; add it to .env")
            sdk = OpenAI(base_url=config.base_url, api_key=key, timeout=config.timeout, max_retries=0)
            transport = lambda request: sdk.chat.completions.create(**request).model_dump()
        self.transport = transport

    def _request(self, prompt, *, system_prompt=None, history=None, max_tokens=None, scoring=False, temperature=0,
                 response_format=None):
        if max_tokens is not None and max_tokens > self.config.max_tokens:
            raise ValueError("Requested output exceeds configured token budget")
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.extend(history or [])
        # AutoToM sends its likelihood prompts as a system message; we do the same when scoring.
        messages.append({"role": "system" if scoring else "user", "content": prompt})
        tokens = self.config.scoring_max_tokens if scoring else (max_tokens or self.config.max_tokens)
        request = dict(model=self.config.model, messages=messages, seed=self.config.seed,
                       temperature=0 if scoring else temperature, **{self.config.token_param: tokens},
                       **json.loads(json.dumps(self.config.extra)))
        if scoring:
            request.update(logprobs=True, top_logprobs=self.config.top_logprobs)
        if response_format is not None:
            request["response_format"] = response_format
        for attempt in range(self.config.max_retries + 1):
            if len(self.records) >= self.config.max_calls:
                raise RuntimeError("Model call budget exhausted")
            started = time.monotonic()
            record = {"request": request, "config": asdict(self.config), "attempt": attempt}
            try:
                reply = self.transport(request)
                record.update(response=reply, status="ok")
                return reply
            except Exception as error:
                record.update(status="error", error_type=type(error).__name__)
                if not isinstance(error, TRANSIENT) or attempt == self.config.max_retries:
                    raise
                time.sleep(2 ** attempt)
            finally:
                record["seconds"] = time.monotonic() - started
                self.records.append(record)
                if self.log_path:
                    self.log_path.parent.mkdir(parents=True, exist_ok=True)
                    with self.log_path.open("a") as stream:
                        stream.write(json.dumps(record) + "\n")

    def generate(self, prompt, *, system_prompt=None, history=None, max_tokens=None, temperature=0,
                 response_format=None, allow_truncated=False):
        """allow_truncated: return text cut off at max_tokens, as upstream code does with its own
        OpenAI client (ThoughtTracing caps likelihood reasoning at 512 tokens and parses whatever
        came back). Each such reply is counted as `truncated` in usage()."""
        reply = self._request(prompt, system_prompt=system_prompt, history=history, max_tokens=max_tokens,
                              temperature=temperature, response_format=response_format)
        choice = reply["choices"][0]
        text = choice["message"].get("content")
        reasoning = choice["message"].get("reasoning_content") or choice["message"].get("reasoning")
        finish = choice.get("finish_reason")
        if finish == "length" and allow_truncated and isinstance(text, str) and text.strip():
            self.records[-1]["truncated"] = True
        elif finish != "stop" or not isinstance(text, str) or not text.strip():
            raise ProtocolError("Missing, truncated, or non-text response")
        if "<think>" in text or "</think>" in text or reasoning:
            raise ProtocolError("Unexpected thinking output in non-thinking mode")
        return text.strip()

    def score_label(self, prompt, label="A", fallback=None):
        """P(A) under the agreed rule: P(A) if the model wrote A, 1 - P(B) if it wrote B.

        If the first token is neither letter (e.g. "The"), we do what AutoToM does with GPT-4o:
        use A's logprob if A is among the returned alternatives, and otherwise return
        `fallback` (AutoToM uses 0.1). Without a fallback that case is an error. Every fallback
        is counted in usage()["score_fallbacks"] and marked in the call log.
        """
        if label != "A":
            raise ValueError("Only P(A) for an A/B question is supported")
        reply = self._request(prompt, scoring=True)
        try:
            first = reply["choices"][0]["logprobs"]["content"][0]
            token, logprob = first["token"].strip(), first["logprob"]
        except (KeyError, IndexError, TypeError) as error:
            raise ProtocolError("Server did not return a logprob for the answer token") from error
        if not math.isfinite(logprob) or logprob > 1e-9:
            raise ProtocolError("Invalid answer logprob")
        probability = min(math.exp(logprob), 1.0)
        if token == "A":
            return probability
        if token == "B":
            return 1 - probability
        for alternative in first.get("top_logprobs") or []:
            if alternative.get("token", "").strip() == "A" and math.isfinite(alternative.get("logprob", math.nan)):
                self._note_fallback("top_logprobs", token)
                return min(math.exp(alternative["logprob"]), 1.0)
        if fallback is None:
            raise ProtocolError(f"Expected the answer A or B, got {token!r}")
        self._note_fallback("constant", token)
        return fallback

    def _note_fallback(self, kind, token):
        self.records[-1]["score_fallback"] = kind
        if self.log_path:
            with self.log_path.open("a") as stream:
                stream.write(json.dumps({"score_fallback": kind, "first_token": token,
                                         "call": len(self.records) - 1}) + "\n")

    def usage(self):
        usage = {"calls": len(self.records), "prompt_tokens": 0, "completion_tokens": 0,
                 "seconds": sum(record["seconds"] for record in self.records), "usage_missing": 0,
                 "score_fallbacks": sum("score_fallback" in record for record in self.records),
                 "truncated": sum(bool(record.get("truncated")) for record in self.records)}
        for record in self.records:
            reported = record.get("response", {}).get("usage")
            if not reported:
                usage["usage_missing"] += 1
                continue
            for key in ("prompt_tokens", "completion_tokens"):
                usage[key] += reported.get(key) or 0
        return usage
