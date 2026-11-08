"""Answer prompts, copied from ThoughtTracing's own evaluation scripts for each benchmark.

The plain baseline sends ThoughtTracing's no-tracing prompt; the ThoughtTracing arm sends the
same prompt with the trace inserted before its "Therefore" line. The tracer input itself is
built by the loader (Example.native["thoughttracing"]["input"]). ExploreToM has no released
prompt, so it uses the ToMi one, whose location question it shares.
"""

# revised_tomi/test_tomi.py
TOMI_ANSWER = ("The short one-sentence answer specifying the most detailed location including both "
               "the container and the place (e.g., from A in B) without any explanation is:")
TOMI_THEREFORE = "Therefore, " + TOMI_ANSWER[0].lower() + TOMI_ANSWER[1:]
# bigtom/test_bigtom.py and mmtom_qa/test_mmtom.py
THEREFORE = "Therefore, the answer is:"
MMTOM_RESPOND = "Please respond with either (a) or (b) without any explanation."
# Output budgets in those scripts: BigToM 128 tokens; ToMi 256; MMToM-QA the agent default.
MAX_TOKENS = {"bigtom": 128, "tomi": 256, "mmtom": 1024}


def plain_prompt(dataset, tt_input):
    if dataset == "tomi":
        return tt_input.replace("Answer:", TOMI_ANSWER)
    if dataset in ("bigtom", "mmtom"):
        return tt_input
    raise ValueError(f"No answer prompt for {dataset}")


def traced_prompt(dataset, tt_input, trace):
    if dataset == "tomi":
        return tt_input.removesuffix("\nAnswer:") + f"\n\n{trace.strip()}\n\n{TOMI_THEREFORE}"
    if dataset == "bigtom":
        return tt_input.removesuffix("\nAnswer:") + f"\n\n{trace}\n{THEREFORE}"
    if dataset == "mmtom":
        stem = tt_input.replace(MMTOM_RESPOND, "").replace("\nAnswer:", "").strip()
        return f"{stem}\n\n{trace.strip()}\n\n{MMTOM_RESPOND}\n{THEREFORE}"
    raise ValueError(f"No answer prompt for {dataset}")
