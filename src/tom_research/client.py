"""Small OpenAI-compatible client with explicit budgets and raw token scores."""

from dataclasses import asdict, dataclass
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
    model: str = "Qwen/Qwen3.8-27B"
    revision: str | None = None
    base_url: str = "http://127.0.0.1:8000/v1"
    seed: int = 42
    max_tokens: int = 4096
    timeout: float = 180
    max_retries: int = 2
    max_calls: int = 500
    top_logprobs: int = 20


class LLMClient:
    def __init__(self, config, transport=None, log_path=None):
        self.config = config
        self.records = []
        self.log_path = Path(log_path) if log_path else None
        if transport is None:
            sdk = OpenAI(base_url=config.base_url, api_key=os.getenv("OPENAI_API_KEY", "local"),
                         timeout=config.timeout, max_retries=0)
            transport = lambda request: sdk.chat.completions.create(**request).model_dump()
        self.transport = transport

    def _request(self, prompt, *, system_prompt=None, history=None, max_tokens=None, scoring=False, temperature=0):
        if max_tokens is not None and max_tokens > self.config.max_tokens:
            raise ValueError("Requested output exceeds configured token budget")
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.extend(history or [])
        messages.append({"role": "system" if scoring else "user", "content": prompt})
        request = dict(
            model=self.config.model, messages=messages, temperature=0 if scoring else temperature, top_p=1,
            seed=self.config.seed, max_tokens=1 if scoring else (max_tokens or self.config.max_tokens),
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        if scoring:
            request.update(logprobs=True, top_logprobs=self.config.top_logprobs)
        transient = (APIConnectionError, APITimeoutError, InternalServerError, RateLimitError)
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
                if not isinstance(error, transient) or attempt == self.config.max_retries:
                    raise
            finally:
                record["seconds"] = time.monotonic() - started
                self.records.append(record)
                if self.log_path:
                    self.log_path.parent.mkdir(parents=True, exist_ok=True)
                    with self.log_path.open("a") as stream:
                        stream.write(json.dumps(record) + "\n")

    def generate(self, prompt, *, system_prompt=None, history=None, max_tokens=None, temperature=0):
        reply = self._request(prompt, system_prompt=system_prompt, history=history, max_tokens=max_tokens, temperature=temperature)
        choice = reply["choices"][0]
        text = choice["message"].get("content")
        reasoning = choice["message"].get("reasoning_content") or choice["message"].get("reasoning")
        if choice.get("finish_reason") != "stop" or not isinstance(text, str) or not text.strip():
            raise ProtocolError("Missing, truncated, or non-text response")
        if "<think>" in text or "</think>" in text or reasoning:
            raise ProtocolError("Unexpected thinking output in non-thinking mode")
        return text.strip()

    def score_label(self, prompt, label="A"):
        reply = self._request(prompt, scoring=True)
        try:
            scores = reply["choices"][0]["logprobs"]["content"][0]["top_logprobs"]
        except (KeyError, IndexError, TypeError) as error:
            raise ProtocolError("Server did not return label logprobs") from error
        for score in scores:
            if score.get("bytes") == list(label.encode()) and score.get("token") == label:
                logprob = score["logprob"]
                if not math.isfinite(logprob) or logprob > 0:
                    raise ProtocolError("Invalid label logprob")
                return math.exp(logprob)
        raise ProtocolError(f"Exact label {label!r} absent from top logprobs; no fallback probability used")

    def usage(self):
        usage = {"calls": len(self.records), "prompt_tokens": 0, "completion_tokens": 0,
                 "seconds": sum(record["seconds"] for record in self.records), "usage_missing": 0}
        for record in self.records:
            reported = record.get("response", {}).get("usage")
            if not reported:
                usage["usage_missing"] += 1
                continue
            for key in ("prompt_tokens", "completion_tokens"):
                usage[key] += reported.get(key) or 0
        return usage