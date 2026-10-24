"""Try ONE real LLM attempt at a generated problem and see what happens.

    python -m experiments.try_llm [seed] [preset]

    seed    any integer (default 3)
    preset  easy | medium | hard (default medium)

Prints the prompt, the model's reply, what the checker thinks of it, the
feedback prompt that would be sent next if the schedule was wrong, and the
solver's plan for comparison. It makes at most one API request (and none if
the same prompt was already cached).
"""

import sys

from scheduler.checker import check_plan
from scheduler.generator import generate_preset
from scheduler.llm import LLMConfigError, make_llm_from_env
from scheduler.parsing import parse_reply
from scheduler.problem import format_plan
from scheduler.prompts import build_feedback_prompt, build_plan_prompt
from scheduler.solver import solve


def main() -> int:
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    preset = sys.argv[2] if len(sys.argv) > 2 else "medium"

    problem, _ = generate_preset(preset, seed)
    prompt = build_plan_prompt(problem)
    print(f"=== PROMPT ({preset}, seed {seed}) ===")
    print(prompt)

    try:
        llm = make_llm_from_env()
    except LLMConfigError as error:
        print(f"\nSetup problem: {error}")
        return 1

    result = llm.complete(prompt)
    source = "from cache" if result.cached else f"from the API, {result.latency:.1f}s"
    print(f"\n=== MODEL REPLY ({source}) ===")
    if not result.ok:
        print("ERROR:", result.error)
        return 1
    print(result.text.strip())

    print("\n=== CHECK ===")
    parsed = parse_reply(result.text)
    if not parsed.ok:
        print("The reply could not be read:")
        messages = list(parsed.errors)
    else:
        verdict = check_plan(problem, parsed.plan)
        if verdict.valid:
            print("VALID schedule!")
            messages = []
        else:
            print(f"INVALID, {len(verdict.violations)} problem(s):")
            messages = verdict.messages()
    for message in messages:
        print(f"  - {message}")

    if messages:
        print("\n=== FEEDBACK PROMPT (what the next step would send) ===")
        print(build_feedback_prompt(problem, result.text, parsed.plan, messages))

    print("\n=== FOR COMPARISON: the solver's plan ===")
    print(format_plan(solve(problem).plan))
    return 0


if __name__ == "__main__":
    sys.exit(main())