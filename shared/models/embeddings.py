"""Embeddings for Experiment 5 (near-duplicate hypotheses), logged like chat calls."""

import json
import math
import os
from pathlib import Path
import time

from openai import OpenAI

from shared.models.providers import EMBEDDING_MODEL


class EmbeddingClient:
    def __init__(self, model=EMBEDDING_MODEL["model"], base_url=EMBEDDING_MODEL["base_url"],
                 key_env=EMBEDDING_MODEL["key_env"], transport=None, log_path=None, max_calls=500):
        self.model, self.max_calls = model, max_calls
        self.records = []
        self.log_path = Path(log_path) if log_path else None
        if transport is None:
            key = os.getenv(key_env)
            if not key:
                raise RuntimeError(f"{key_env} is not set; add it to .env")
            sdk = OpenAI(base_url=base_url, api_key=key, max_retries=2)
            transport = lambda request: sdk.embeddings.create(**request).model_dump()
        self.transport = transport

    def embed(self, texts):
        if len(self.records) >= self.max_calls:
            raise RuntimeError("Embedding call budget exhausted")
        request = {"model": self.model, "input": list(texts), "encoding_format": "float"}
        started = time.monotonic()
        reply = self.transport(request)
        record = {"request": request, "usage": reply.get("usage"), "seconds": time.monotonic() - started}
        self.records.append(record)
        if self.log_path:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.log_path.open("a") as stream:
                stream.write(json.dumps(record) + "\n")
        vectors = [item["embedding"] for item in sorted(reply["data"], key=lambda item: item["index"])]
        if len(vectors) != len(request["input"]):
            raise RuntimeError("Embedding count does not match input count")
        return vectors


def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    return dot / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)))
