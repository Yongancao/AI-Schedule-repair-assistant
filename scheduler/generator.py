"""Random problem generator and "change" generator for the experiments.

Generating problems
-------------------
Every problem is built around a *planted* valid schedule: tasks are first
placed on the timeline, and only then are deadlines, earliest-start times and
prerequisites derived from that placement. That guarantees each generated
problem is solvable, so when a method fails we know it is the method's fault,
not an impossible request. (Changes, below, can optionally produce
unsolvable problems to test infeasibility detection.)

Difficulty is controlled by a few knobs:
    n_tasks         how many tasks
    n_blocks        how many fixed events split the day
    utilization     fraction of free time that tasks fill (0.9 = very tight)
    deadline_prob   chance a task has a deadline
    deadline_slack  max minutes of slack between planted end and deadline
    dep_prob        chance a task has prerequisites

Generating changes
------------------
A *change* is something that happens after a plan exists: a new meeting,
a new task, a longer task, a shorter day, a tighter deadline. Repair methods
must produce a new valid plan. `make_change` produces changes that really
break the old plan (otherwise there is nothing to repair).

All randomness comes from `random.Random(seed)`, so every problem and change
is reproducible from its seed.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, replace
from typing import Optional

from .checker import check_plan
from .problem import Block, Problem, Task, fmt_time
from .solver import earliest_fit, solve

GRID = 15  # all generated times and durations are multiples of 15 minutes

TASK_NAMES = [
    "write_report", "gym", "groceries", "laundry", "study_ml", "email_backlog",
    "call_mom", "cook_dinner", "clean_kitchen", "pay_bills", "read_paper",
    "debug_project", "meal_prep", "walk_dog", "plan_trip", "review_notes",
    "water_plants", "update_resume", "book_flights", "tidy_desk",
    "practice_guitar", "fix_bike", "pack_boxes", "organize_files",
    "refill_prescription", "wash_car", "submit_form", "draft_slides",
    "back_up_laptop", "buy_gift",
]

BLOCK_LABELS = [
    "team_meeting", "dentist", "lunch_with_sam", "office_hours",
    "doctor_visit", "interview", "class", "client_call", "haircut",
]

PRESETS = {
    "easy": dict(n_tasks=4, n_blocks=1, utilization=0.5,
                 deadline_prob=0.4, deadline_slack=90, dep_prob=0.2),
    "medium": dict(n_tasks=7, n_blocks=2, utilization=0.75,
                   deadline_prob=0.6, deadline_slack=45, dep_prob=0.3),
    "hard": dict(n_tasks=10, n_blocks=2, utilization=0.9,
                 deadline_prob=0.8, deadline_slack=15, dep_prob=0.4),
}

CHANGE_KINDS = (
    "new_block", "add_task", "extend_task", "shrink_window", "tighten_deadline",
)


class GenerationError(RuntimeError):
    """Raised when a valid problem or change could not be generated."""


@dataclass(frozen=True)
class Change:
    """Something that happened after the plan was made."""

    kind: str
    description: str      # plain English, e.g. for an LLM repair prompt
    problem: Problem      # the problem after the change


# --------------------------------------------------------------------------
# Problem generation
# --------------------------------------------------------------------------

def generate_with_solution(
    seed: int,
    n_tasks: int = 6,
    n_blocks: int = 1,
    utilization: float = 0.7,
    deadline_prob: float = 0.6,
    deadline_slack: int = 60,
    dep_prob: float = 0.3,
    earliest_start_prob: float = 0.15,
    day_start: int = 8 * 60,
    day_end: int = 18 * 60,
    max_attempts: int = 500,
) -> tuple:
    """Return (problem, planted_plan); the planted plan is a valid solution."""
    rng = random.Random(seed)
    for _ in range(max_attempts):
        result = _try_generate(
            rng, n_tasks, n_blocks, utilization, deadline_prob,
            deadline_slack, dep_prob, earliest_start_prob, day_start, day_end,
        )
        if result is not None:
            problem, plan = result
            if not check_plan(problem, plan).valid:  # internal sanity check
                raise GenerationError("Planted plan failed the checker.")
            return problem, plan
    raise GenerationError(
        f"Could not generate a problem after {max_attempts} attempts; "
        "try lower utilization, fewer tasks, or fewer blocks."
    )


def generate_problem(seed: int, **knobs) -> Problem:
    """Like generate_with_solution but returns only the problem."""
    return generate_with_solution(seed, **knobs)[0]


def generate_preset(name: str, seed: int) -> tuple:
    """(problem, planted_plan) for the "easy", "medium" or "hard" preset."""
    return generate_with_solution(seed, **PRESETS[name])


def _snap(value: float) -> int:
    """Round to the nearest multiple of GRID."""
    return int(round(value / GRID)) * GRID


def _try_generate(
    rng, n_tasks, n_blocks, utilization, deadline_prob, deadline_slack,
    dep_prob, earliest_start_prob, day_start, day_end,
) -> Optional[tuple]:
    # 1. Fixed events that do not overlap each other.
    blocks = []
    labels = rng.sample(BLOCK_LABELS, min(n_blocks, len(BLOCK_LABELS)))
    for label in labels:
        length = rng.choice([30, 45, 60, 90])
        placed = False
        for _ in range(50):
            latest_start = day_end - length
            if latest_start < day_start:
                break
            start = day_start + GRID * rng.randint(
                0, (latest_start - day_start) // GRID
            )
            end = start + length
            if all(end <= b.start or start >= b.end for b in blocks):
                blocks.append(Block(start, end, label))
                placed = True
                break
        if not placed:
            return None
    blocks.sort(key=lambda b: b.start)

    skeleton = Problem(day_start, day_end, (), tuple(blocks))
    free = skeleton.free_intervals()
    free_total = sum(end - start for start, end in free)
    longest_gap = max((end - start for start, end in free), default=0)
    if longest_gap < GRID:
        return None

    # 2. Task durations that add up to roughly utilization * free time.
    target = utilization * free_total
    weights = [rng.uniform(0.5, 1.5) for _ in range(n_tasks)]
    scale = target / sum(weights)
    max_duration = (longest_gap // GRID) * GRID
    durations = [
        min(max(GRID, _snap(weight * scale)), max_duration)
        for weight in weights
    ]
    if sum(durations) > free_total:
        return None

    # 3. Names, a hidden planting order, and prerequisites that respect it.
    names = _task_names(rng, n_tasks)
    order = rng.sample(range(n_tasks), n_tasks)  # order[p] = task index
    depends = {i: [] for i in range(n_tasks)}
    for position in range(1, n_tasks):
        if rng.random() < dep_prob:
            earlier = order[:position]
            how_many = 2 if position >= 2 and rng.random() < 0.3 else 1
            for prerequisite in rng.sample(earlier, how_many):
                depends[order[position]].append(names[prerequisite])

    # 4. Plant a valid schedule, left to right, with random idle gaps.
    slack_left = free_total - sum(durations)
    cursor = day_start
    starts = {}
    for i in order:
        gap = 0
        if slack_left > 0 and rng.random() < 0.5:
            gap = min(slack_left, rng.choice([15, 30, 45, 60]))
        start = earliest_fit(free, durations[i], cursor + gap)
        if start is None:
            return None
        slack_left -= gap
        starts[i] = start
        cursor = start + durations[i]

    # 5. Derive deadlines / earliest starts from the planted schedule.
    slack_choices = list(range(0, max(deadline_slack, 0) + 1, GRID))
    tasks = []
    for i in range(n_tasks):
        end = starts[i] + durations[i]
        deadline = None
        if rng.random() < deadline_prob:
            deadline = min(day_end, end + rng.choice(slack_choices))
        earliest_start = None
        if rng.random() < earliest_start_prob:
            candidate = starts[i] - rng.choice(slack_choices)
            if candidate > day_start:
                earliest_start = candidate
        tasks.append(Task(
            id=names[i],
            duration=durations[i],
            deadline=deadline,
            earliest_start=earliest_start,
            depends_on=tuple(depends[i]),
        ))

    # Hide the planting order: list the tasks in random order.
    rng.shuffle(tasks)
    plan = [
        {"task_id": names[i], "start": starts[i],
         "end": starts[i] + durations[i]}
        for i in sorted(range(n_tasks), key=lambda k: starts[k])
    ]
    return Problem(day_start, day_end, tuple(tasks), tuple(blocks)), plan


def _task_names(rng: random.Random, count: int) -> list:
    if count <= len(TASK_NAMES):
        return rng.sample(TASK_NAMES, count)
    names = list(TASK_NAMES)
    rng.shuffle(names)
    extra = [f"task_{n}" for n in range(1, count - len(TASK_NAMES) + 1)]
    return names + extra


# --------------------------------------------------------------------------
# Change generation
# --------------------------------------------------------------------------

def make_change(
    problem: Problem,
    plan: list,
    seed: int,
    kind: Optional[str] = None,
    require_feasible: bool = True,
    require_breaking: bool = True,
    attempts: int = 200,
) -> Change:
    """Create a change to `problem` that happens after `plan` was made.

    require_breaking  the old plan must be invalid after the change, so
                      there is something to repair
    require_feasible  the changed problem must still have a valid plan
    kind              one of CHANGE_KINDS, or None to pick at random
    """
    if kind is not None and kind not in CHANGE_KINDS:
        raise ValueError(f"Unknown change kind {kind!r}.")

    rng = random.Random(seed)
    for _ in range(attempts):
        chosen = kind or rng.choice(CHANGE_KINDS)
        proposal = _PROPOSERS[chosen](problem, plan, rng)
        if proposal is None:
            continue
        changed, description = proposal
        if require_breaking and check_plan(changed, plan).valid:
            continue
        if require_feasible and not solve(changed).solved:
            continue
        return Change(chosen, description, changed)

    raise GenerationError(
        f"Could not make a {kind or 'suitable'} change after "
        f"{attempts} attempts."
    )


def _propose_new_block(problem, plan, rng):
    entry = rng.choice(plan)
    length = rng.choice([30, 45, 60])
    start = _snap(entry["start"] + rng.choice([-15, 0, 15, 30]))
    end = start + length
    if start < problem.day_start or end > problem.day_end:
        return None
    if any(start < b.end and b.start < end for b in problem.blocks):
        return None
    used = {b.label for b in problem.blocks}
    labels = [label for label in BLOCK_LABELS if label not in used]
    if not labels:
        return None
    label = rng.choice(labels)
    changed = replace(
        problem, blocks=problem.blocks + (Block(start, end, label),)
    )
    return changed, (
        f"A new fixed event was added: {label} from "
        f"{fmt_time(start)} to {fmt_time(end)}."
    )


def _propose_add_task(problem, plan, rng):
    used = set(problem.task_ids())
    names = [name for name in TASK_NAMES if name not in used]
    if not names:
        return None
    name = rng.choice(names)
    duration = rng.choice([15, 30, 45, 60])

    deadline = None
    if rng.random() < 0.5:
        latest = (problem.day_end // GRID) * GRID
        earliest = problem.day_start + duration
        if latest < earliest:
            return None
        deadline = _snap(rng.uniform(earliest, latest))
        deadline = min(max(deadline, earliest), latest)

    depends_on = ()
    if rng.random() < 0.3:
        depends_on = (rng.choice(problem.task_ids()),)

    new_task = Task(name, duration, deadline=deadline, depends_on=depends_on)
    changed = replace(problem, tasks=problem.tasks + (new_task,))

    text = f"A new task was added: '{name}' needs {duration} minutes"
    if deadline is not None:
        text += f" and must be done by {fmt_time(deadline)}"
    if depends_on:
        text += f" and must come after '{depends_on[0]}'"
    return changed, text + "."


def _propose_extend_task(problem, plan, rng):
    task = rng.choice(problem.tasks)
    new_duration = task.duration + rng.choice([15, 30, 45])
    tasks = tuple(
        replace(t, duration=new_duration) if t.id == task.id else t
        for t in problem.tasks
    )
    return replace(problem, tasks=tasks), (
        f"Task '{task.id}' now needs {new_duration} minutes "
        f"instead of {task.duration}."
    )


def _propose_shrink_window(problem, plan, rng):
    new_end = problem.day_end - rng.choice([30, 60, 90, 120])
    if new_end <= problem.day_start:
        return None
    return replace(problem, day_end=new_end), (
        f"Available hours now end at {fmt_time(new_end)} "
        f"instead of {fmt_time(problem.day_end)}."
    )


def _propose_tighten_deadline(problem, plan, rng):
    entry = rng.choice(plan)
    task = problem.task(entry["task_id"])
    new_deadline = _snap(entry["end"] - rng.choice([15, 30, 60]))
    if new_deadline < problem.day_start + task.duration:
        return None
    tasks = tuple(
        replace(t, deadline=new_deadline) if t.id == task.id else t
        for t in problem.tasks
    )
    return replace(problem, tasks=tasks), (
        f"Task '{task.id}' must now be finished by {fmt_time(new_deadline)}."
    )


_PROPOSERS = {
    "new_block": _propose_new_block,
    "add_task": _propose_add_task,
    "extend_task": _propose_extend_task,
    "shrink_window": _propose_shrink_window,
    "tighten_deadline": _propose_tighten_deadline,
}
