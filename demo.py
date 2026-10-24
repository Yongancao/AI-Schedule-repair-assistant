"""Quick tour of the scheduling core. Run:  python demo.py [seed] [preset]

    seed    any integer (default 3)
    preset  easy | medium | hard (default medium)

Shows: a generated problem, the solver's plan, a change that breaks the plan,
what the checker says, and how a minimal-change repair compares with simply
re-solving the whole day from scratch.
"""

import sys

from scheduler.checker import check_plan
from scheduler.generator import generate_preset, make_change
from scheduler.problem import format_plan, format_problem, moved_tasks, total_shift
from scheduler.solver import repair, solve


def main() -> None:
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    preset = sys.argv[2] if len(sys.argv) > 2 else "medium"

    problem, _ = generate_preset(preset, seed)
    print(f"=== Problem ({preset}, seed {seed}) ===")
    print(format_problem(problem))

    result = solve(problem)
    plan = result.plan
    print(f"\n=== Solver plan ({result.nodes} search nodes) ===")
    print(format_plan(plan))
    print("Checker says valid:", check_plan(problem, plan).valid)

    change = make_change(problem, plan, seed)
    print(f"\n=== Change ({change.kind}) ===")
    print(change.description)

    broken = check_plan(change.problem, plan)
    print("\nOld plan is now invalid:")
    for message in broken.messages():
        print("  -", message)

    fixed = repair(change.problem, plan)
    scratch = solve(change.problem)

    print(f"\n=== Minimal-change repair: moved {fixed.moved} task(s) ===")
    print(format_plan(fixed.plan))
    print("Checker says valid:", check_plan(change.problem, fixed.plan).valid)
    print("Moved:", moved_tasks(plan, fixed.plan) or "none",
          "| total shift:", total_shift(plan, fixed.plan), "min")

    print("\n=== For comparison: re-solve from scratch ===")
    print(format_plan(scratch.plan))
    print("Moved:", len(moved_tasks(plan, scratch.plan)), "task(s) |",
          "total shift:", total_shift(plan, scratch.plan), "min")


if __name__ == "__main__":
    main()
