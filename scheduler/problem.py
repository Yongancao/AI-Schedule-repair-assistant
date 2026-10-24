"""Problem and plan data structures for the single-day scheduling domain.

Everything in the project shares these definitions: the checker, the solver,
the problem generator, the LLM prompts, and (later) the UI.

Conventions
-----------
* Every time is an integer number of minutes after midnight (13:00 -> 780).
* A *Problem* describes one person's day: an availability window, fixed
  events ("blocks") that cannot be moved, and tasks that must be scheduled.
* A *plan* is a list of dicts, one per task:
      {"task_id": "gym", "start": 780, "end": 840}
  Plans stay as plain dicts because that is exactly what an LLM returns as
  JSON, and the checker must be able to inspect untrusted data.
"""

from __future__ import annotations

from dataclasses import dataclass
from graphlib import CycleError, TopologicalSorter
from typing import Any, Optional

MINUTES_PER_DAY = 24 * 60

# A plan is a list of {"task_id": str, "start": int, "end": int} dicts.
Plan = list


class ProblemError(ValueError):
    """Raised when a problem description is malformed."""


# --------------------------------------------------------------------------
# Time helpers
# --------------------------------------------------------------------------

def fmt_time(minutes: int) -> str:
    """Convert minutes after midnight to 'HH:MM' (1440 -> '24:00')."""
    hours, mins = divmod(minutes, 60)
    return f"{hours:02d}:{mins:02d}"


def parse_time(text: str) -> int:
    """Convert a 24-hour 'HH:MM' string to minutes after midnight."""
    try:
        hours_text, minutes_text = text.strip().split(":")
        hours, minutes = int(hours_text), int(minutes_text)
    except (ValueError, AttributeError):
        raise ProblemError(f"Time {text!r} must look like HH:MM.") from None

    in_range = 0 <= minutes <= 59 and (
        0 <= hours <= 23 or (hours == 24 and minutes == 0)
    )
    if not in_range:
        raise ProblemError(f"Time {text!r} is outside 00:00-24:00.")
    return hours * 60 + minutes


def _is_int(value: Any) -> bool:
    """True for real integers; False for bools, floats, strings."""
    return isinstance(value, int) and not isinstance(value, bool)


# --------------------------------------------------------------------------
# Problem definition
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Task:
    """Something that must be done in one continuous block.

    duration        exact length in minutes
    deadline        the task must END at or before this time (optional)
    earliest_start  the task must START at or after this time (optional)
    depends_on      ids of tasks that must FINISH before this one starts
    """

    id: str
    duration: int
    deadline: Optional[int] = None
    earliest_start: Optional[int] = None
    depends_on: tuple = ()

    def __post_init__(self):
        object.__setattr__(self, "depends_on", tuple(self.depends_on))


@dataclass(frozen=True)
class Block:
    """A fixed event (meeting, appointment) that tasks cannot overlap."""

    start: int
    end: int
    label: str = ""


@dataclass(frozen=True)
class Problem:
    """One day to schedule: availability window, fixed events, and tasks."""

    day_start: int
    day_end: int
    tasks: tuple
    blocks: tuple = ()

    def __post_init__(self):
        object.__setattr__(self, "tasks", tuple(self.tasks))
        object.__setattr__(self, "blocks", tuple(self.blocks))
        errors = self.validate()
        if errors:
            raise ProblemError("; ".join(errors))

    # -- validation --------------------------------------------------------

    def validate(self) -> list:
        """Return a list of human-readable problems (empty list = valid)."""
        errors = []

        if not (_is_int(self.day_start) and _is_int(self.day_end)):
            return ["day_start and day_end must be integers."]
        if not 0 <= self.day_start < self.day_end <= MINUTES_PER_DAY:
            errors.append("Need 0 <= day_start < day_end <= 1440.")

        for block in self.blocks:
            if not (_is_int(block.start) and _is_int(block.end)):
                errors.append("Block times must be integers.")
            elif not 0 <= block.start < block.end <= MINUTES_PER_DAY:
                errors.append(
                    f"Block {block.label or ''!r} needs "
                    "0 <= start < end <= 1440."
                )

        ids = [task.id for task in self.tasks]
        for task in self.tasks:
            if not isinstance(task.id, str) or not task.id.strip():
                errors.append("Every task needs a non-empty text id.")
                continue
            if ids.count(task.id) > 1:
                message = f"Task id {task.id!r} is used more than once."
                if message not in errors:
                    errors.append(message)
            if not _is_int(task.duration) or not (
                1 <= task.duration <= MINUTES_PER_DAY
            ):
                errors.append(f"Task {task.id!r}: duration must be 1-1440.")
            if task.deadline is not None and not (
                _is_int(task.deadline) and 0 < task.deadline <= MINUTES_PER_DAY
            ):
                errors.append(f"Task {task.id!r}: deadline must be 1-1440.")
            if task.earliest_start is not None and not (
                _is_int(task.earliest_start)
                and 0 <= task.earliest_start < MINUTES_PER_DAY
            ):
                errors.append(f"Task {task.id!r}: earliest_start must be 0-1439.")
            for dep in task.depends_on:
                if dep == task.id:
                    errors.append(f"Task {task.id!r} cannot depend on itself.")
                elif dep not in ids:
                    errors.append(
                        f"Task {task.id!r} depends on unknown task {dep!r}."
                    )

        if not errors:
            graph = {task.id: set(task.depends_on) for task in self.tasks}
            try:
                tuple(TopologicalSorter(graph).static_order())
            except CycleError:
                errors.append("Task dependencies contain a cycle.")

        return errors

    # -- convenience -------------------------------------------------------

    def task(self, task_id: str) -> Task:
        for task in self.tasks:
            if task.id == task_id:
                return task
        raise KeyError(task_id)

    def task_ids(self) -> list:
        return [task.id for task in self.tasks]

    def total_duration(self) -> int:
        return sum(task.duration for task in self.tasks)

    def free_intervals(self) -> list:
        """The availability window minus fixed events, as (start, end) gaps."""
        free = []
        cursor = self.day_start
        for block in sorted(self.blocks, key=lambda b: (b.start, b.end)):
            start = max(block.start, self.day_start)
            end = min(block.end, self.day_end)
            if end <= start:
                continue  # block lies outside the window
            if start > cursor:
                free.append((cursor, start))
            cursor = max(cursor, end)
        if cursor < self.day_end:
            free.append((cursor, self.day_end))
        return free

    def free_minutes(self) -> int:
        return sum(end - start for start, end in self.free_intervals())

    # -- JSON round trip ---------------------------------------------------

    def to_dict(self) -> dict:
        """Plain-dict form (JSON-serializable); optional fields are omitted."""
        tasks = []
        for task in self.tasks:
            entry = {"id": task.id, "duration": task.duration}
            if task.deadline is not None:
                entry["deadline"] = task.deadline
            if task.earliest_start is not None:
                entry["earliest_start"] = task.earliest_start
            if task.depends_on:
                entry["depends_on"] = list(task.depends_on)
            tasks.append(entry)

        blocks = []
        for block in self.blocks:
            entry = {"start": block.start, "end": block.end}
            if block.label:
                entry["label"] = block.label
            blocks.append(entry)

        return {
            "day_start": self.day_start,
            "day_end": self.day_end,
            "blocks": blocks,
            "tasks": tasks,
        }

    @classmethod
    def from_dict(cls, data: Any) -> "Problem":
        """Build a Problem from a dict; raises ProblemError if malformed."""
        if not isinstance(data, dict):
            raise ProblemError("A problem must be a JSON object.")
        try:
            tasks = tuple(
                Task(
                    id=item["id"],
                    duration=item["duration"],
                    deadline=item.get("deadline"),
                    earliest_start=item.get("earliest_start"),
                    depends_on=tuple(item.get("depends_on") or ()),
                )
                for item in data["tasks"]
            )
            blocks = tuple(
                Block(
                    start=item["start"],
                    end=item["end"],
                    label=item.get("label", ""),
                )
                for item in data.get("blocks") or ()
            )
            return cls(
                day_start=data["day_start"],
                day_end=data["day_end"],
                tasks=tasks,
                blocks=blocks,
            )
        except (KeyError, TypeError, AttributeError) as error:
            raise ProblemError(f"Malformed problem: {error!r}") from error


# --------------------------------------------------------------------------
# Plan helpers
# --------------------------------------------------------------------------

def sort_plan(plan: Plan) -> Plan:
    """Return the plan ordered by start time."""
    return sorted(plan, key=lambda e: (e["start"], e["end"], e["task_id"]))


def format_plan(plan: Plan) -> str:
    """Readable multi-line schedule, one task per line."""
    return "\n".join(
        f"{fmt_time(e['start'])}-{fmt_time(e['end'])}  {e['task_id']}"
        for e in sort_plan(plan)
    )


def format_problem(problem: Problem) -> str:
    """Readable multi-line summary of a problem."""
    lines = [
        f"Available: {fmt_time(problem.day_start)}-{fmt_time(problem.day_end)}"
    ]
    for block in problem.blocks:
        label = block.label or "fixed event"
        lines.append(
            f"Fixed event: {label} {fmt_time(block.start)}-{fmt_time(block.end)}"
        )
    for task in problem.tasks:
        parts = [f"{task.id}: {task.duration} min"]
        if task.earliest_start is not None:
            parts.append(f"not before {fmt_time(task.earliest_start)}")
        if task.deadline is not None:
            parts.append(f"done by {fmt_time(task.deadline)}")
        if task.depends_on:
            parts.append("after " + ", ".join(task.depends_on))
        lines.append("Task " + ", ".join(parts))
    return "\n".join(lines)


def moved_tasks(old_plan: Plan, new_plan: Plan) -> list:
    """Ids of tasks present in both plans whose start time changed."""
    old_starts = {e["task_id"]: e["start"] for e in old_plan}
    return [
        e["task_id"]
        for e in new_plan
        if e["task_id"] in old_starts and old_starts[e["task_id"]] != e["start"]
    ]


def total_shift(old_plan: Plan, new_plan: Plan) -> int:
    """Total minutes by which tasks present in both plans were shifted."""
    old_starts = {e["task_id"]: e["start"] for e in old_plan}
    return sum(
        abs(e["start"] - old_starts[e["task_id"]])
        for e in new_plan
        if e["task_id"] in old_starts
    )
