"""Exact state-space search for scheduling and for minimal-change repair.

Why a hand-written solver instead of a library?
    Scheduling with deadlines, fixed events and prerequisites is a classic
    state-space search problem, so this file is the course's "search" topic
    applied directly. The instances in this project are small (<= ~15 tasks),
    where exact search is fast, and the result is verified against brute
    force in tests/test_solver.py.

How it works
------------
A schedule on one person's day is a *sequence* of tasks. Given the sequence,
the best way to place each task is as early as possible after the previous
one ("left shift"), because finishing earlier never hurts any later
constraint. So the search is over sequences:

    state  = (set of tasks already scheduled, time when the last one ended)
    action = pick an unscheduled task whose prerequisites are done, place it

Two ideas keep the search small:
    * Dominance: if we have already reached the same set of tasks with an
      earlier-or-equal end time (and no more moved tasks), the new state is
      worse and is pruned.
    * Branch and bound (repair only): abandon a branch once it already moves
      as many tasks as the best complete plan found so far.

Repair
------
`repair` takes a changed problem plus the plan that used to be valid, and
finds a valid plan that changes the START TIME of as few tasks as possible.
Each task has at most two candidate placements at each step:
    * its old start time (cost 0), if that still works, or
    * the earliest feasible start (cost 1).
This is enough to be optimal for the number of moved tasks: in any optimal
plan, every task is either at its old start or can be left-shifted without
changing the count. (Verified against brute force in the tests.)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .problem import Problem

DEFAULT_MAX_NODES = 2_000_000


@dataclass(frozen=True)
class SolveResult:
    """Outcome of a search.

    status
        "solved"      a valid plan was found (optimal when repairing)
        "infeasible"  the search proved no valid plan exists
        "limit"       the node limit was hit before finishing; `plan` holds
                      the best plan found so far, or None
    plan    list of {"task_id", "start", "end"} sorted by start, or None
    moved   number of tasks whose start differs from the old plan
            (always 0 for a plain solve, which has no old plan)
    nodes   search nodes expanded (a rough measure of effort)
    """

    status: str
    plan: Optional[list]
    moved: Optional[int]
    nodes: int

    @property
    def solved(self) -> bool:
        return self.status == "solved"


# --------------------------------------------------------------------------
# Placement helpers
# --------------------------------------------------------------------------

def earliest_fit(free: list, duration: int, lower_bound: int) -> Optional[int]:
    """Earliest start >= lower_bound where `duration` minutes fit in a gap."""
    for gap_start, gap_end in free:
        start = max(gap_start, lower_bound)
        if start + duration <= gap_end:
            return start
    return None


def fits_at(free: list, duration: int, start: int) -> bool:
    """True if [start, start + duration) lies inside a single free gap."""
    return any(
        gap_start <= start and start + duration <= gap_end
        for gap_start, gap_end in free
    )


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------

def solve(problem: Problem, max_nodes: int = DEFAULT_MAX_NODES) -> SolveResult:
    """Find any valid plan, preferring earliest-deadline-first sequences."""
    return _search(problem, {}, max_nodes)


def repair(
    problem: Problem,
    old_plan: list,
    max_nodes: int = DEFAULT_MAX_NODES,
) -> SolveResult:
    """Find a valid plan that moves as few tasks as possible.

    `problem` is the CHANGED problem; `old_plan` is the plan that was valid
    before the change. Tasks in the old plan that no longer exist are
    ignored; tasks that are new have no old start and are never "moved".
    """
    known = set(problem.task_ids())
    old_starts = {
        entry["task_id"]: entry["start"]
        for entry in old_plan
        if entry["task_id"] in known
    }
    return _search(problem, old_starts, max_nodes)


# --------------------------------------------------------------------------
# Search
# --------------------------------------------------------------------------

def _search(problem: Problem, old_starts: dict, max_nodes: int) -> SolveResult:
    tasks = list(problem.tasks)
    count = len(tasks)
    index = {task.id: i for i, task in enumerate(tasks)}

    # prerequisite_mask[i] has a bit set for every task that must finish
    # before task i can start.
    prerequisite_mask = [0] * count
    for i, task in enumerate(tasks):
        for dependency in task.depends_on:
            prerequisite_mask[i] |= 1 << index[dependency]

    free = problem.free_intervals()
    full = (1 << count) - 1
    old_start = [old_starts.get(task.id) for task in tasks]

    # Order in which candidate tasks are tried. Without an old plan this is
    # earliest-deadline-first; with one, tasks are tried in their old order
    # so the first complete plan found is usually already close to optimal.
    def priority(i: int) -> tuple:
        task = tasks[i]
        deadline = task.deadline if task.deadline is not None else 10**9
        return (
            old_start[i] is None,
            old_start[i] or 0,
            deadline,
            task.duration,
            task.id,
        )

    order = sorted(range(count), key=priority)

    # Without an old plan every valid plan is equally good, so stop at the
    # first one. With an old plan, search on until the best is proven.
    first_only = not old_starts

    best_plan = None
    best_moved = count + 1
    nodes = 0
    hit_limit = False
    stop = False
    seen = {}      # mask -> list of (end_time, moved) already explored
    placed = []    # (task index, start) pairs for the current partial plan

    def dominated(mask: int, cursor: int, moved: int) -> bool:
        return any(
            seen_cursor <= cursor and seen_moved <= moved
            for seen_cursor, seen_moved in seen.get(mask, ())
        )

    def remember(mask: int, cursor: int, moved: int) -> None:
        frontier = seen.setdefault(mask, [])
        frontier[:] = [
            (c, m) for c, m in frontier if not (cursor <= c and moved <= m)
        ]
        frontier.append((cursor, moved))

    def dfs(mask: int, cursor: int, moved: int) -> None:
        nonlocal best_plan, best_moved, nodes, hit_limit, stop

        if stop:
            return
        nodes += 1
        if nodes > max_nodes:
            hit_limit = True
            stop = True
            return

        if mask == full:
            if moved < best_moved:
                best_moved = moved
                best_plan = list(placed)
                if first_only or moved == 0:
                    stop = True
            return

        if moved >= best_moved or dominated(mask, cursor, moved):
            return
        remember(mask, cursor, moved)

        for i in order:
            if (mask >> i) & 1 or prerequisite_mask[i] & ~mask:
                continue

            task = tasks[i]
            lower = max(cursor, task.earliest_start or 0)
            earliest = earliest_fit(free, task.duration, lower)
            if earliest is None:
                continue
            if task.deadline is not None and earliest + task.duration > task.deadline:
                continue  # even the earliest placement misses the deadline

            candidates = []
            old = old_start[i]
            if (
                old is not None
                and old >= lower
                and fits_at(free, task.duration, old)
                and (task.deadline is None or old + task.duration <= task.deadline)
            ):
                candidates.append(old)          # keep the old start: cost 0
            if earliest != old:
                candidates.append(earliest)     # left-shifted: cost 1 if old exists

            for start in candidates:
                cost = 0 if (old is None or start == old) else 1
                placed.append((i, start))
                dfs(mask | (1 << i), start + task.duration, moved + cost)
                placed.pop()
                if stop:
                    return

    dfs(0, problem.day_start, 0)

    plan = None
    if best_plan is not None:
        plan = [
            {
                "task_id": tasks[i].id,
                "start": start,
                "end": start + tasks[i].duration,
            }
            for i, start in best_plan
        ]

    if hit_limit:
        status = "limit"
    elif plan is None:
        status = "infeasible"
    else:
        status = "solved"

    return SolveResult(
        status=status,
        plan=plan,
        moved=best_moved if plan is not None else None,
        nodes=nodes,
    )
