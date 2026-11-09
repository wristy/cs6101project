"""Probe whether an API gives us a usable P(A) for AutoToM-style A/B scoring.

For each model it answers three questions:
  1. How many alternative tokens come back with top_logprobs (1 = only the chosen one)?
  2. Does logit_bias on the "A"/"B" tokens work at all (does it force A/B output)?
  3. Is the reported logprob computed AFTER the bias (exact P(A)/(P(A)+P(B))) or BEFORE it?

It then prints which tier the model falls in:
  best   - both letters visible, or reported logprobs reflect the bias -> exact P(A)
  middle - only the chosen letter, unaffected by bias -> P(A) or 1 - P(B) workaround
  worst  - no logprobs at all -> AutoToM cannot run on this model as designed

Usage (keys come from the environment, never from the command line):
  DEEPINFRA_API_KEY=... OPENAI_API_KEY=... python logprob_probe.py
  python logprob_probe.py --only mimo --a-id 32 --b-id 33   # override token IDs
Results are also written to probe_results.json next to this script.
"""

import argparse
import json
import math
import os
from pathlib import Path

from openai import OpenAI

MODELS = {
    "mimo": dict(base_url="https://api.deepinfra.com/v1/openai", key_env="DEEPINFRA_API_KEY",
                 model="XiaomiMiMo/MiMo-V2.6-Pro",
                 extra=dict(extra_body={"chat_template_kwargs": {"enable_thinking": False}}, temperature=0),
                 tokenizer="hf:XiaomiMiMo/MiMo-V2.6-Pro-RL"),  # API model id has no HF repo; same tokenizer
    "gpt6sol": dict(base_url="https://api.openai.com/v1", key_env="OPENAI_API_KEY", model="gpt-6-sol",
                    extra=dict(reasoning_effort="none", temperature=0), tokenizer="tiktoken:o200k_base",
                    max_tokens_param="max_completion_tokens", top_logprobs=5,
                    max_tokens_value=16),  # API caps top_logprobs at 5; rejects max_completion_tokens=1
}

# A realistic AutoToM-style likelihood question (the model should answer A).
AB_PROMPT = ("Story: Sally puts the apple in the basket and leaves the room. Anne moves the apple to the box.\n"
             "Sally believes the apple is in the basket.\n"
             "Question: Given her belief, how likely is it that Sally looks for the apple in the basket?\n"
             "A) Likely\nB) Unlikely\nAnswer with a single letter.")
# A question whose natural answer is NOT A or B; only a working bias can force A/B here.
FORCED_PROMPT = "What is the capital of France? Answer with one word."


def token_ids(spec, override_a, override_b):
    if override_a is not None and override_b is not None:
        return {"A": override_a, "B": override_b}
    kind, name = spec.split(":", 1)
    try:
        if kind == "tiktoken":
            import tiktoken
            enc = tiktoken.get_encoding(name)
            encode = enc.encode
        else:
            from tokenizers import Tokenizer
            tok = Tokenizer.from_pretrained(name)
            encode = lambda s: tok.encode(s, add_special_tokens=False).ids
        ids = {s: encode(s) for s in ("A", "B", " A", " B")}
        return {s: i[0] for s, i in ids.items() if len(i) == 1}
    except Exception as error:  # tokenizer not reachable: probe still runs without the bias tests
        print(f"  ! could not load tokenizer {spec}: {error}. Pass --a-id/--b-id to test logit_bias.")
        return {}


def call(client, cfg, prompt, bias=None):
    request = dict(model=cfg["model"], messages=[{"role": "user", "content": prompt}],
                   logprobs=True, top_logprobs=cfg.get("top_logprobs", 20), seed=42, **cfg["extra"])
    request[cfg.get("max_tokens_param", "max_tokens")] = cfg.get("max_tokens_value", 1)
    if bias:
        request["logit_bias"] = bias
    try:
        reply = client.chat.completions.create(**request).model_dump()
    except Exception as error:
        return {"error": f"{type(error).__name__}: {error}"[:400]}
    choice = reply["choices"][0]
    content = (choice.get("logprobs") or {}).get("content") or []
    if not content:
        return {"text": choice["message"].get("content"), "logprobs": None}
    first = content[0]
    tops = {t["token"]: t["logprob"] for t in first.get("top_logprobs") or []}
    return {"text": choice["message"].get("content"), "token": first["token"], "logprob": first["logprob"],
            "p_chosen": math.exp(first["logprob"]), "n_alternatives": len(tops), "top": tops}


def letter_mass(result):
    top = result.get("top") or {}
    return sum(math.exp(lp) for tok, lp in top.items() if tok.strip() in ("A", "B"))


def probe(name, cfg, a_id, b_id):
    print(f"\n=== {name} ({cfg['model']}) ===")
    key = os.getenv(cfg["key_env"])
    if not key:
        print(f"  ! {cfg['key_env']} not set; skipped")
        return {"skipped": True}
    client = OpenAI(base_url=cfg["base_url"], api_key=key)
    ids = token_ids(cfg["tokenizer"], a_id, b_id)
    bias = {str(i): 100 for i in ids.values()} if ids else None
    out = {"token_ids": ids}
    out["ab_plain"] = call(client, cfg, AB_PROMPT)
    out["forced_plain"] = call(client, cfg, FORCED_PROMPT)
    if bias:
        out["ab_bias"] = call(client, cfg, AB_PROMPT, bias)
        out["forced_bias"] = call(client, cfg, FORCED_PROMPT, bias)
    for label, result in out.items():
        if label != "token_ids":
            print(f"  {label:13s} {json.dumps({k: v for k, v in result.items() if k != 'top'})}")

    plain = out["ab_plain"]
    if plain.get("error") or plain.get("logprobs", 1) is None:
        tier, why = "worst", "no logprobs returned (or the request was rejected)"
    elif plain.get("n_alternatives", 0) >= 2 and letter_mass(plain) > 0 and \
            {t.strip() for t in plain["top"]} >= {"A", "B"}:
        tier, why = "best", "both A and B are visible in top_logprobs"
    else:
        tier, why = "middle", "only the chosen token is visible"
        forced = out.get("forced_bias") or {}
        if forced.get("token", "").strip() in ("A", "B"):
            # Bias forced A/B on a question whose natural answer is "Paris". If the reported
            # logprob were pre-bias, it would be tiny (the model never wanted "A"); a value
            # near log(0.5) or above means the API reports the post-bias distribution.
            if forced["logprob"] > math.log(0.2):
                tier, why = "best", "logit_bias works and reported logprobs are post-bias"
            else:
                why += "; logit_bias works but reported logprobs are pre-bias"
        elif bias:
            why += "; logit_bias did not force A/B (wrong token IDs, or bias ignored)"
    out["tier"], out["why"] = tier, why
    print(f"  -> tier: {tier.upper()} ({why})")
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", choices=list(MODELS))
    parser.add_argument("--a-id", type=int)
    parser.add_argument("--b-id", type=int)
    args = parser.parse_args()
    results = {name: probe(name, cfg, args.a_id, args.b_id)
               for name, cfg in MODELS.items() if args.only in (None, name)}
    path = Path(__file__).with_name("probe_results.json")
    path.write_text(json.dumps(results, indent=2))
    print(f"\nSaved {path}")


if __name__ == "__main__":
    main()
