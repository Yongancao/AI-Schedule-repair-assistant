"""Plan checker: decides whether a plan satisfies a problem, and why not.

The checker is the project's ground truth. Every method (LLM-only, LLM with
feedback, solver, hybrid, repair) is scored by this one function, so it has
to be strict and independent of the solver.

Design notes
------------
* A plan is untrusted data (it may come straight from an LLM), so the
  structure is validated first. If the structure is broken, only structure
  errors are reported, because nothing else can be checked safely.
* Violations carry a plain-English message. Those messages are exactly what
  the "LLM + checker feedback" method sends back to the model.
* Each constraint is reported once, in one place. A task with an invalid
  time range is not also reported as overlapping everything else.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .problem import MINUTES_PER_DAY, Problem, _is_int, fmt_time

# Every violation type the checker can produce (useful for error analysis).
VIOLATION_TYPES = (
    "invalid_structure",
    "missing_task",
    "unknown_task",
    "duplicate_task",
    "bad_time_range",
    "wrong_duration",
    "outside_availability",
    "before_earliest_start",
    "missed_deadline",
    "overlap",
    "blocked_conflict",
    "dependency_violation",
)


@dataclass(frozen=True)
class Violation:
    type: str
    message: str
    task_ids: tuple = ()


@dataclass(frozen=True)
class CheckResult:
    violations: tuple

    @property
    def valid(self) -> bool:
        return not self.violations

    def types(self) -> list:
        return [v.type for v in self.violations]

    def messages(self) -> list:
        return [v.message for v in self.violations]

    def counts(self) -> dict:
        """How many violations of each type (for error analysis)."""
        counts = {}
        for violation in self.violations:
            counts[violation.type] = counts.get(violation.type, 0) + 1
        return counts


def check_plan(problem: Problem, plan: Any) -> CheckResult:
    """Check `plan` against `problem` and return every violation found."""
    structure_errors = _check_structure(plan)
    if structure_errors:
        return CheckResult(tuple(structure_errors))

    violations = []
    violations += _check_coverage(problem, plan)
    violations += _check_each_entry(problem, plan)

    usable = [entry for entry in plan if _range_ok(entry)]
    violations += _check_overlaps(usable)
    violations += _check_blocks(problem, usable)
    violations += _check_dependencies(problem, plan)

    return CheckResult(tuple(violations))


# --------------------------------------------------------------------------
# Individual checks
# --------------------------------------------------------------------------

def _range_ok(entry: dict) -> bool:
    return 0 <= entry["start"] < entry["end"] <= MINUTES_PER_DAY


def _check_structure(plan: Any) -> list:
    if not isinstance(plan, list):
        return [Violation(
            "invalid_structure",
            "The plan must be a JSON list of entries.",
        )]

    errors = []
    for number, entry in enumerate(plan, start=1):
        label = f"Entry {number}"
        if not isinstance(entry, dict):
            errors.append(Violation(
                "invalid_structure", f"{label} must be an object."
            ))
            continue

        task_id = entry.get("task_id")
        if not isinstance(task_id, str) or not task_id.strip():
            errors.append(Violation(
                "invalid_structure",
                f"{label} needs a non-empty text 'task_id'.",
            ))
        for field in ("start", "end"):
            if not _is_int(entry.get(field)):
                errors.append(Violation(
                    "invalid_structure",
                    f"{label} needs an integer '{field}' "
                    "(minutes after midnight).",
                ))
    return errors


def _check_coverage(problem: Problem, plan: list) -> list:
    counts = {}
    for entry in plan:
        counts[entry["task_id"]] = counts.get(entry["task_id"], 0) + 1

    violations = []
    known = set(problem.task_ids())

    for task in problem.tasks:
        if task.id not in counts:
            violations.append(Violation(
                "missing_task",
                f"Required task '{task.id}' is missing from the plan.",
                (task.id,),
            ))

    for task_id, count in counts.items():
        if task_id not in known:
            violations.append(Violation(
                "unknown_task",
                f"Task '{task_id}' is not one of the required tasks.",
                (task_id,),
            ))
        elif count > 1:
            violations.append(Violation(
                "duplicate_task",
                f"Task '{task_id}' appears {count} times; "
                "it must appear exactly once.",
                (task_id,),
            ))
    return violations


def _check_each_entry(problem: Problem, plan: list) -> list:
    violations = []
    tasks = {task.id: task for task in problem.tasks}

    for entry in plan:
        task_id = entry["task_id"]
        start, end = entry["start"], entry["end"]

        if not _range_ok(entry):
            violations.append(Violation(
                "bad_time_range",
                f"Task '{task_id}' has an invalid time range "
                f"({start}-{end}); it needs 0 <= start < end <= 1440.",
                (task_id,),
            ))
            continue

        task = tasks.get(task_id)
        if task is None:
            continue  # reported as unknown_task

        scheduled = end - start
        if scheduled != task.duration:
            violations.append(Violation(
                "wrong_duration",
                f"Task '{task_id}' needs exactly {task.duration} minutes, "
                f"but {scheduled} minutes are scheduled "
                f"({fmt_time(start)}-{fmt_time(end)}).",
                (task_id,),
            ))

        if start < problem.day_start or end > problem.day_end:
            violations.append(Violation(
                "outside_availability",
                f"Task '{task_id}' ({fmt_time(start)}-{fmt_time(end)}) is "
                f"outside available hours "
                f"{fmt_time(problem.day_start)}-{fmt_time(problem.day_end)}.",
                (task_id,),
            ))

        if task.earliest_start is not None and start < task.earliest_start:
            violations.append(Violation(
                "before_earliest_start",
                f"Task '{task_id}' starts at {fmt_time(start)}, before its "
                f"earliest allowed start of {fmt_time(task.earliest_start)}.",
                (task_id,),
            ))

        if task.deadline is not None and end > task.deadline:
            violations.append(Violation(
                "missed_deadline",
                f"Task '{task_id}' ends at {fmt_time(end)}, after its "
                f"deadline of {fmt_time(task.deadline)}.",
                (task_id,),
            ))
    return violations


def _check_overlaps(entries: list) -> list:
    violations = []
    for i in range(len(entries)):
        for j in range(i + 1, len(entries)):
            first, second = entries[i], entries[j]
            start = max(first["start"], second["start"])
            end = min(first["end"], second["end"])
            if start < end:
                violations.append(Violation(
                    "overlap",
                    f"Tasks '{first['task_id']}' and '{second['task_id']}' "
                    f"overlap from {fmt_time(start)} to {fmt_time(end)} "
                    f"({end - start} minutes).",
                    (first["task_id"], second["task_id"]),
                ))
    return violations


def _check_blocks(problem: Problem, entries: list) -> list:
    violations = []
    for entry in entries:
        for block in problem.blocks:
            start = max(entry["start"], block.start)
            end = min(entry["end"], block.end)
            if start < end:
                label = block.label or "fixed event"
                violations.append(Violation(
                    "blocked_conflict",
                    f"Task '{entry['task_id']}' "
                    f"({fmt_time(entry['start'])}-{fmt_time(entry['end'])}) "
                    f"conflicts with the fixed event '{label}' "
                    f"({fmt_time(block.start)}-{fmt_time(block.end)}).",
                    (entry["task_id"],),
                ))
    return violations


def _check_dependencies(problem: Problem, plan: list) -> list:
    entries = {}
    for entry in plan:
        entries.setdefault(entry["task_id"], []).append(entry)

    violations = []
    for task in problem.tasks:
        current = entries.get(task.id, [])
        if len(current) != 1 or not _range_ok(current[0]):
            continue  # coverage / time-range checks already report this

        for prerequisite_id in task.depends_on:
            prerequisite = entries.get(prerequisite_id, [])
            if len(prerequisite) != 1 or not _range_ok(prerequisite[0]):
                continue

            if prerequisite[0]["end"] > current[0]["start"]:
                violations.append(Violation(
                    "dependency_violation",
                    f"Task '{task.id}' starts at "
                    f"{fmt_time(current[0]['start'])}, but its prerequisite "
                    f"'{prerequisite_id}' does not finish until "
                    f"{fmt_time(prerequisite[0]['end'])}.",
                    (task.id, prerequisite_id),
                ))
    return violations
