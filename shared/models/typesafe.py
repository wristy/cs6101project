"""TypeSafe's Jev ("System One") API: probabilities without generating text.

POST https://api.typesafe.ai/v1/systemone with a `state` (what the questions are about) and
named `questions`. We use two question types (docs.typesafe.ai/api):
- noul:  {"type": "noul", "instructions": ...} -> {"noul": P(yes/true)}           (Experiment 4)
- choice: {"type": "choice", "instructions": ..., "criteria": {name: description}}
          -> {"choice": name, "confidence", "probabilities": {name: p}}          (Experiment 2a)
The API has no seed or temperature. We pin a model version so runs are reproducible as far as
the API allows; the model name the server reports is logged with every call.
"""

import json
import os
from pathlib import Path
import time

import requests

URL = "https://api.typesafe.ai/v1/systemone"
MODEL = "jev-1.13.0"
TRANSIENT_STATUS = {429, 500, 502, 503, 529}


class JevClient:
    def __init__(self, model=MODEL, key_env="JEV_API_KEY", transport=None, log_path=None,
                 max_calls=500, timeout=60, max_retries=2):
        self.model, self.max_calls, self.max_retries = model, max_calls, max_retries
        self.records = []
        self.log_path = Path(log_path) if log_path else None
        if transport is None:
            key = os.getenv(key_env)
            if not key:
                raise RuntimeError(f"{key_env} is not set; add it to .env")
            session = requests.Session()
            session.headers["Authorization"] = f"Bearer {key}"

            def transport(request):
                try:
                    response = session.post(URL, json=request, timeout=timeout)
                except (requests.ConnectionError, requests.Timeout) as error:
                    raise TransientError(str(error)) from error
                if response.status_code in TRANSIENT_STATUS:
                    raise TransientError(f"TypeSafe returned {response.status_code}")
                if response.status_code != 200:
                    raise RuntimeError(f"TypeSafe returned {response.status_code}: {response.text[:500]}")
                return response.json()
        self.transport = transport

    def ask(self, state, questions):
        """Send one request; returns the answers dict keyed by question name."""
        request = {"model": self.model, "state": state, "questions": questions}
        for attempt in range(self.max_retries + 1):
            if len(self.records) >= self.max_calls:
                raise RuntimeError("TypeSafe call budget exhausted")
            started = time.monotonic()
            record = {"request": request, "attempt": attempt, "provider": "typesafe"}
            try:
                reply = self.transport(request)
                record.update(response=reply, status="ok")
                answers = reply["answers"]
                if set(answers) != set(questions):
                    raise ValueError("TypeSafe answered different questions than asked")
                return answers
            except TransientError:
                record.update(status="error", error_type="TransientError")
                if attempt == self.max_retries:
                    raise
                time.sleep(2 ** attempt)
            finally:
                record["seconds"] = time.monotonic() - started
                self.records.append(record)
                if self.log_path:
                    self.log_path.parent.mkdir(parents=True, exist_ok=True)
                    with self.log_path.open("a") as stream:
                        stream.write(json.dumps(record) + "\n")

    def noul(self, state, statement):
        """P(statement is true | state)."""
        answer = self.ask(state, {"q": {"type": "noul", "instructions": statement}})["q"]
        probability = answer.get("noul")
        if answer.get("type") != "noul" or not isinstance(probability, (int, float)) or not 0 <= probability <= 1:
            raise ValueError(f"Invalid noul answer: {answer}")
        return float(probability)

    def choice(self, state, instructions, options):
        """Probability over unordered named options ({name: description}); returns the full answer."""
        options = dict(options)
        answer = self.ask(state, {"q": {"type": "choice", "instructions": instructions,
                                        "criteria": options}})["q"]
        probabilities = answer.get("probabilities")
        if (answer.get("type") != "choice" or not isinstance(probabilities, dict)
                or set(probabilities) != set(options) or answer.get("choice") not in options):
            raise ValueError(f"Invalid choice answer: {answer}")
        return answer

    def usage(self):
        return {"jev_calls": len(self.records),
                "jev_input_tokens": sum((r.get("response") or {}).get("usage", {}).get("input_tokens", 0)
                                        for r in self.records),
                "jev_models": sorted({(r.get("response") or {}).get("model") for r in self.records} - {None})}


class TransientError(RuntimeError):
    pass
