"""Prompt text for the LLM methods.

Two prompts:

    build_plan_prompt      "Here is a day to schedule. Reply with a schedule."
    build_feedback_prompt  The same prompt, followed by the model's previous
                           answer and the checker's list of problems.

Design choices
--------------
* Times appear as 24-hour HH:MM, not minutes after midnight. That is how a
  person (and a model) naturally reasons about a day, and it keeps the test
  fair: the model is not being tripped up by an unusual number format.
* The rules are spelled out in full. The experiment measures whether the
  model can *do* the scheduling, not whether it can guess the rules.
* The feedback prompt contains only the latest attempt and its problems, not
  the whole history. It stays short, and the first prompt is identical to the
  "LLM only" prompt, so those two methods share one cached reply.
"""

from __future__ import annotations

from typing import Optional

from .parsing import format_plan_json
from .problem import Problem, fmt_time

RULES = """\
Rules:
1. Every task must appear exactly once, as one continuous block that lasts exactly its stated duration.
2. Tasks cannot overlap each other or the fixed events. A task may start at the exact minute another ends.
3. Every task must lie inside the available hours.
4. Respect each task's deadline, earliest start time, and prerequisites. A task with prerequisites may only start after all of them have finished.
All times use 24-hour HH:MM format."""

OUTPUT_FORMAT = """\
Reply with ONLY a JSON list, with no explanation, using the task ids exactly as given. Example of the format:
[{"task_id": "example_task", "start": "09:00", "end": "09:30"}]"""

MAX_PREVIOUS_REPLY_CHARS = 1500


def problem_to_text(problem: Problem) -> str:
    """The problem as a readable list, with times in HH:MM."""
    lines = [
        f"Available hours: {fmt_time(problem.day_start)} to "
        f"{fmt_time(problem.day_end)}",
    ]

    if problem.blocks:
        lines.append("Fixed events (no task may overlap these):")
        for block in sorted(problem.blocks, key=lambda b: (b.start, b.end)):
            label = block.label or "fixed event"
            lines.append(
                f"  - {label}: {fmt_time(block.start)} to {fmt_time(block.end)}"
            )
    else:
        lines.append("Fixed events: none")

    lines.append("Tasks to schedule:")
    for task in problem.tasks:
        details = [f"{task.duration} minutes"]
        if task.earliest_start is not None:
            details.append(f"cannot start before {fmt_time(task.earliest_start)}")
        if task.deadline is not None:
            details.append(f"must finish by {fmt_time(task.deadline)}")
        if task.depends_on:
            details.append(
                "can only start after these have finished: "
                + ", ".join(task.depends_on)
            )
        lines.append(f"  - {task.id}: " + "; ".join(details))

    return "\n".join(lines)


def build_plan_prompt(problem: Problem) -> str:
    return (
        "You are a careful scheduling assistant. "
        "Build a schedule for one person's day.\n\n"
        f"{problem_to_text(problem)}\n\n"
        f"{RULES}\n\n"
        f"{OUTPUT_FORMAT}"
    )


def build_feedback_prompt(
    problem: Problem,
    previous_reply: str,
    previous_plan: Optional[list],
    messages: list,
) -> str:
    """Original prompt + the last attempt + what the checker found wrong.

    If the last reply could be read, it is shown in clean JSON form;
    otherwise the raw reply is shown (shortened if very long).
    """
    if previous_plan is not None:
        shown = format_plan_json(previous_plan)
    else:
        shown = previous_reply.strip()
        if len(shown) > MAX_PREVIOUS_REPLY_CHARS:
            shown = shown[:MAX_PREVIOUS_REPLY_CHARS] + " ... (shortened)"

    problems = "\n".join(f"- {message}" for message in messages)
    return (
        f"{build_plan_prompt(problem)}\n\n"
        f"Your previous answer was:\n{shown}\n\n"
        f"A checker found these problems with it:\n{problems}\n\n"
        "Fix every problem. Reply with the COMPLETE corrected schedule "
        "(every task, not just the changed ones) in the same JSON format, "
        "with no explanation."
    )