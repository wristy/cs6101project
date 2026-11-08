"""Do closed models give *graded* P(A) on real AutoToM likelihood prompts?

Sends 20 AutoToM-style likelihood prompts (AutoToM's own templates from model/probs.py,
filled with BigToM stories from ThoughtTracing's 249-question set) to each model, exactly
as AutoToM does: prompt as a system message, temperature 0, top_logprobs requested, one-letter
answer. Records the chosen letter and its probability, then summarises the spread.

If P(chosen) sits at ~1.0 on nearly every prompt, AutoToM's weighting collapses into hard votes.
If it varies (e.g. many values between 0.5 and 0.99), the P(A) / 1 - P(B) workaround is usable.

Usage: OPENAI_API_KEY=... DEEPINFRA_API_KEY=... python autotom_likelihood_probe.py
Writes autotom_likelihood_results.json next to this script.
"""

import json
import math
import os
import statistics
from pathlib import Path

from openai import OpenAI

OPENAI = dict(base_url="https://api.openai.com/v1", key_env="OPENAI_API_KEY",
              extra=dict(reasoning_effort="none", temperature=0, max_completion_tokens=16))
MODELS = {
    "gpt-6-sol": OPENAI,
    "gpt-6.1-terra": OPENAI,
    "gpt-6-luna": OPENAI,
    # Reference: MiMo on DeepInfra, to confirm its graded values hold on real prompts.
    "XiaomiMiMo/MiMo-V2.6-Pro": dict(base_url="https://api.deepinfra.com/v1/openai", key_env="DEEPINFRA_API_KEY",
                                     extra=dict(temperature=0, max_tokens=1,
                                                extra_body={"chat_template_kwargs": {"enable_thinking": False}})),
}

# --- AutoToM templates (verbatim from AutoToM model/probs.py, get_likelihood_general) ---
def belief_given_observation(info, agent, statement):
    return f"""Determine if the statement is likely, respond with only either A or B.
{info}
Here is a statement of {agent}'s current belief. If {agent}'s current belief is not aligned with {agent}'s observation, it is very unlikely.
Determine if the following statement is likely: {statement}
A) Likely.
B) Unlikely."""

def action_given_goal_belief(info, agent, statement):
    return f"""Determine if the statement is likely, respond with only either A or B.
{info}
Here is a statement of {agent}'s action. The belief stands for {agent}'s current belief which is true. {agent} is likely to act according to goal and belief concerning certain objects (the wording for objects must be same. You should ignore the correlation of different objects. e.g., plate and apple are two different objects.) Notice that {agent}'s belief does not represent the goal.
When belief and goal are irrelevant, and action is directly driven by goal, it's likely. When belief and goal are relevant (about exactly the same object) and they contradict (or irrelevant) with action, it's unlikely.
Determine if the following statement is likely: {statement}
A) Likely.
B) Unlikely."""

def general(info, agent, statement):
    return f"""Determine if the statement is likely, respond with only either A or B.  If it's not certain but it's possible, it's considered likely. If it contradicts to the given information in some way, then it is unlikely. 
{info}
Determine if the following statement is likely: {statement}
A) Likely.
B) Unlikely."""

# --- 20 prompts from 5 BigToM stories; some clear, some deliberately ambiguous ---
CASES = [
    # Noor, false belief (did not see swap)
    (belief_given_observation, "Noor", "State: The milk pitcher contains almond milk.\nNoor's observation: Noor filled the pitcher with oat milk. Noor did not see her coworker swapping the milk.",
     ["Noor believes the milk pitcher contains oat milk.", "Noor believes the milk pitcher contains almond milk."]),
    (action_given_goal_belief, "Noor", "Noor's goal: make a cappuccino with oat milk for the customer.\nNoor's belief: the milk pitcher contains oat milk.",
     ["Noor makes the cappuccino using the milk in the pitcher.", "Noor opens the fridge to take out the oat milk again."]),
    # Noor, true belief (saw swap)
    (belief_given_observation, "Noor", "State: The milk pitcher contains almond milk.\nNoor's observation: Noor sees her coworker swapping the oat milk in the pitcher with almond milk.",
     ["Noor believes the milk pitcher contains almond milk.", "Noor believes the milk pitcher contains oat milk."]),
    (action_given_goal_belief, "Noor", "Noor's goal: make a cappuccino with oat milk for the customer.\nNoor's belief: the milk pitcher contains almond milk.",
     ["Noor replaces the almond milk with oat milk.", "Noor makes the cappuccino using the milk in the pitcher."]),
    # Luka, weather (ambiguous belief update)
    (belief_given_observation, "Luka", "State: The clouds have blown away and the sky is bright and sunny.\nLuka's observation: Luka saw the cloudy sky earlier. It is not stated whether Luka noticed the clouds blowing away.",
     ["Luka believes it will rain soon.", "Luka believes the weather will remain sunny."]),
    (action_given_goal_belief, "Luka", "Luka's goal: water the plants so they stay healthy.\nLuka's belief: it might rain soon.",
     ["Luka waits for the rain to water the plants.", "Luka waters the plants with a hose right away."]),
    # Kofi, soup (false belief)
    (belief_given_observation, "Kofi", "State: The soup contains tomato paste instead of peanut butter.\nKofi's observation: Kofi added peanut butter to the soup. Kofi does not notice the apprentice swapping the ingredients.",
     ["Kofi believes the soup contains peanut butter.", "Kofi believes the soup contains tomato paste."]),
    (action_given_goal_belief, "Kofi", "Kofi's goal: prepare a traditional groundnut soup.\nKofi's belief: the soup contains peanut butter.",
     ["Kofi continues cooking the soup.", "Kofi adds peanut butter back into the soup."]),
    # Kofi, fishing net (true belief) + general-template cases
    (belief_given_observation, "Kofi", "State: The fishing net is torn apart.\nKofi's observation: Kofi observes the monkeys damaging his fishing net.",
     ["Kofi believes his fishing net is torn apart.", "Kofi believes his fishing net is in good condition."]),
    (general, "Kofi", "Kofi is a fisherman who wants to catch enough fish today. While Kofi is away from his boat, monkeys tear his fishing net apart. Kofi observes the monkeys damaging his net.",
     ["Kofi goes fishing with the net as it is.", "Kofi repairs the net before going fishing."]),
]

PROMPTS = [(fn.__name__, statement, fn(info, agent, statement))
           for fn, agent, info, statements in CASES for statement in statements]
assert len(PROMPTS) == 20, len(PROMPTS)


def run(model, cfg):
    key = os.getenv(cfg["key_env"])
    if not key:
        return {"skipped": f"{cfg['key_env']} not set"}
    client = OpenAI(base_url=cfg["base_url"], api_key=key)
    rows = []
    for template, statement, prompt in PROMPTS:
        try:
            reply = client.chat.completions.create(
                model=model, messages=[{"role": "system", "content": prompt}],
                logprobs=True, top_logprobs=5, seed=42, **cfg["extra"]).model_dump()
        except Exception as error:
            rows.append({"statement": statement, "error": f"{type(error).__name__}: {error}"[:300]})
            if len(rows) == 1:  # model unavailable or parameters rejected: stop early
                break
            continue
        content = (reply["choices"][0].get("logprobs") or {}).get("content") or []
        if not content:
            rows.append({"statement": statement, "text": reply["choices"][0]["message"].get("content"), "logprob": None})
            continue
        first = content[0]
        token, p = first["token"].strip(), math.exp(first["logprob"])
        p_a = p if token == "A" else (1 - p if token == "B" else None)  # the agreed workaround
        rows.append({"template": template, "statement": statement, "token": token, "p_chosen": round(p, 6),
                     "p_a": None if p_a is None else round(p_a, 6),
                     "alternatives": {t["token"]: round(math.exp(t["logprob"]), 6) for t in first.get("top_logprobs") or []}})
    ok = [r["p_chosen"] for r in rows if "p_chosen" in r]
    summary = {"answered": len(ok), "errors": sum("error" in r for r in rows)}
    if ok:
        summary.update(min=round(min(ok), 4), median=round(statistics.median(ok), 4), max=round(max(ok), 4),
                       below_0_99=sum(p < 0.99 for p in ok), below_0_9=sum(p < 0.9 for p in ok),
                       letters={l: sum(r.get("token") == l for r in rows) for l in ("A", "B")},
                       verdict="graded" if sum(p < 0.99 for p in ok) >= 4 else "saturated (hard votes)")
    return {"summary": summary, "rows": rows}


def main():
    results = {}
    for model, cfg in MODELS.items():
        print(f"\n=== {model} ===")
        results[model] = run(model, cfg)
        print(json.dumps(results[model].get("summary") or results[model], indent=2))
        for row in results[model].get("rows", []):
            print(f"  {row.get('token', '?'):>2} p={row.get('p_chosen', row.get('error', '-'))}  {row['statement']}")
    path = Path(__file__).with_name("autotom_likelihood_results.json")
    path.write_text(json.dumps(results, indent=2))
    print(f"\nSaved {path}")


if __name__ == "__main__":
    main()
