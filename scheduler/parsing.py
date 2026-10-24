"""Turning an LLM's reply into a plan the checker can judge.

LLMs rarely reply with *only* the JSON you asked for. They add a friendly
sentence, wrap the answer in a ```json fence, or write times a little
differently than requested. This module is the forgiving first step:

    reply text -> find the JSON -> check its shape -> convert "13:00" to 780

It forgives presentation (extra text, code fences, a {"schedule": [...]}
wrapper) but not content: times must be 24-hour "HH:MM" strings, and every
entry must have task_id, start and end. Anything it cannot read becomes a
plain-English error message, which the feedback method sends back to the
model.

Whether the schedule is actually *valid* is not decided here; that is the
checker's job (checker.py).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Optional

from .problem import fmt_time, sort_plan

_FENCE = re.compile(r"```(?:json|JSON)?\s*(.*?)```", re.DOTALL)
_CLOCK = re.compile(r"^\s*(\d{1,2}):(\d{2})\s*$")
_JSON_START = re.compile(r"[\[{]")
_PLAN_KEYS = ("plan", "schedule")


@dataclass(frozen=True)
class ParseResult:
    """plan is a list of {"task_id", "start", "end"} dicts with times in
    minutes, or None if the reply could not be read; errors explains why."""

    plan: Optional[list]
    errors: tuple = ()

    @property
    def ok(self) -> bool:
        return self.plan is not None


# --------------------------------------------------------------------------
# Finding the JSON
# --------------------------------------------------------------------------

def _decode_all(text: str) -> list:
    """Every JSON array/object found in `text`, in order of appearance."""
    decoder = json.JSONDecoder()
    values = []
    position = 0
    while True:
        match = _JSON_START.search(text, position)
        if match is None:
            return values
        try:
            value, end = decoder.raw_decode(text, match.start())
        except ValueError:
            position = match.start() + 1
            continue
        values.append(value)
        position = end


def _looks_like_plan(value: Any) -> bool:
    if isinstance(value, dict):
        return any(isinstance(value.get(key), list) for key in _PLAN_KEYS)
    return isinstance(value, list) and all(isinstance(i, dict) for i in value)


def extract_json(text: Any) -> Any:
    """Find the JSON answer inside a model reply; raises ValueError if none.

    A fenced block is tried first. If the reply contains several JSON values
    (say a stray "[1]" in the explanation), the first one that looks like a
    plan wins.
    """
    if not isinstance(text, str) or not text.strip():
        raise ValueError("the reply is empty")

    sources = []
    fenced = _FENCE.search(text)
    if fenced:
        sources.append(fenced.group(1))
    sources.append(text)

    found = []
    for source in sources:
        for value in _decode_all(source):
            if _looks_like_plan(value):
                return value
            found.append(value)

    if found:
        return found[0]
    raise ValueError("no valid JSON was found in the reply")


# --------------------------------------------------------------------------
# Times
# --------------------------------------------------------------------------

def parse_clock(value: Any) -> int:
    """'13:05' or '9:30' -> minutes after midnight. Raises ValueError."""
    if not isinstance(value, str):
        raise ValueError('it must be a time written as text, like "13:00"')

    match = _CLOCK.match(value)
    if match is None:
        raise ValueError(
            f'{value!r} is not a 24-hour time in HH:MM format, like "13:00"'
        )
    hours, minutes = int(match.group(1)), int(match.group(2))
    if minutes > 59 or hours > 24 or (hours == 24 and minutes != 0):
        raise ValueError(f"{value!r} is not a valid time (use 00:00 to 24:00)")
    return hours * 60 + minutes


# --------------------------------------------------------------------------
# Reply -> plan
# --------------------------------------------------------------------------

def parse_reply(text: Any) -> ParseResult:
    """Read a model reply into a plan (times in minutes) or explain why not."""
    try:
        data = extract_json(text)
    except ValueError as error:
        return ParseResult(None, (f"Your answer could not be read: {error}.",))

    if isinstance(data, dict):
        for key in _PLAN_KEYS:
            if isinstance(data.get(key), list):
                data = data[key]
                break

    if not isinstance(data, list):
        return ParseResult(None, (
            "The answer must be a JSON list of entries like "
            '{"task_id": "name", "start": "HH:MM", "end": "HH:MM"}.',
        ))

    errors = []
    plan = []
    for number, item in enumerate(data, start=1):
        label = f"Entry {number}"
        if not isinstance(item, dict):
            errors.append(f"{label} must be an object with task_id, start and end.")
            continue

        task_id = item.get("task_id")
        if not isinstance(task_id, str) or not task_id.strip():
            errors.append(f"{label} needs a text 'task_id'.")
            continue

        entry = {"task_id": task_id}
        for field in ("start", "end"):
            if field not in item:
                errors.append(f"{label} ('{task_id}') is missing '{field}'.")
                continue
            try:
                entry[field] = parse_clock(item[field])
            except ValueError as problem:
                errors.append(
                    f"{label} ('{task_id}') has a bad '{field}': {problem}."
                )
        if "start" in entry and "end" in entry:
            plan.append(entry)

    if errors:
        return ParseResult(None, tuple(errors))
    return ParseResult(plan, ())


# --------------------------------------------------------------------------
# Plan -> reply text (the reverse direction)
# --------------------------------------------------------------------------

def format_plan_json(plan: list) -> str:
    """A plan as the JSON text we ask models to produce (HH:MM times)."""
    lines = [
        json.dumps({
            "task_id": entry["task_id"],
            "start": fmt_time(entry["start"]),
            "end": fmt_time(entry["end"]),
        })
        for entry in sort_plan(plan)
    ]
    if not lines:
        return "[]"
    return "[\n  " + ",\n  ".join(lines) + "\n]"