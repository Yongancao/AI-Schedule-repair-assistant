import unittest
from itertools import product

from scheduler.checker import check_plan
from scheduler.generator import CHANGE_KINDS, generate_with_solution, make_change
from scheduler.problem import Block, Problem, Task, moved_tasks
from scheduler.solver import earliest_fit, fits_at, repair, solve


def entry(task_id, start, end):
    return {"task_id": task_id, "start": start, "end": end}


def brute_force(problem, old_plan=None, grid=15):
    """Independent reference: try EVERY start-time assignment on the grid.

    Returns (feasible, minimum number of moved tasks). It deliberately uses
    only check_plan, never the solver's own logic.
    """
    options = []
    for task in problem.tasks:
        latest = problem.day_end - task.duration
        options.append([
            (task.id, start, start + task.duration)
            for start in range(problem.day_start, latest + 1, grid)
        ])

    best = None
    for combo in product(*options):
        plan = [entry(*item) for item in combo]
        if not check_plan(problem, plan).valid:
            continue
        moved = len(moved_tasks(old_plan, plan)) if old_plan else 0
        if best is None or moved < best:
            best = moved
            if best == 0:
                break
    return best is not None, best


class PlacementHelpersTest(unittest.TestCase):
    FREE = [(480, 600), (660, 900)]

    def test_earliest_fit(self):
        self.assertEqual(earliest_fit(self.FREE, 60, 0), 480)
        self.assertEqual(earliest_fit(self.FREE, 60, 560), 660)   # 40 min left in gap 1
        self.assertEqual(earliest_fit(self.FREE, 60, 700), 700)
        self.assertEqual(earliest_fit(self.FREE, 300, 0), None)
        self.assertEqual(earliest_fit(self.FREE, 60, 850), None)

    def test_fits_at(self):
        self.assertTrue(fits_at(self.FREE, 60, 480))
        self.assertTrue(fits_at(self.FREE, 60, 540))
        self.assertFalse(fits_at(self.FREE, 60, 570))     # runs into the gap end
        self.assertFalse(fits_at(self.FREE, 120, 540))    # spans the block


class SolveTest(unittest.TestCase):
    def assertSolved(self, problem):
        result = solve(problem)
        self.assertEqual(result.status, "solved")
        self.assertTrue(check_plan(problem, result.plan).valid)
        return result

    def test_simple_problem(self):
        problem = Problem(480, 720, (Task("a", 60), Task("b", 90)))
        self.assertSolved(problem)

    def test_empty_problem(self):
        result = solve(Problem(480, 720, ()))
        self.assertEqual(result.status, "solved")
        self.assertEqual(result.plan, [])

    def test_respects_fixed_events(self):
        problem = Problem(
            480, 720, (Task("a", 90), Task("b", 60)),
            (Block(540, 600, "meeting"),),
        )
        result = self.assertSolved(problem)
        for item in result.plan:
            self.assertFalse(item["start"] < 600 and item["end"] > 540)

    def test_respects_dependencies_and_earliest_start(self):
        problem = Problem(480, 900, (
            Task("a", 60, depends_on=("b",)),
            Task("b", 60, earliest_start=600),
        ))
        result = self.assertSolved(problem)
        starts = {e["task_id"]: e["start"] for e in result.plan}
        self.assertGreaterEqual(starts["b"], 600)
        self.assertGreaterEqual(starts["a"], starts["b"] + 60)

    def test_deadline_forces_order(self):
        problem = Problem(480, 720, (
            Task("late", 60),
            Task("urgent", 60, deadline=540),
        ))
        result = self.assertSolved(problem)
        self.assertEqual(result.plan[0]["task_id"], "urgent")

    def test_task_may_be_placed_after_a_block_when_it_cannot_fit_before(self):
        problem = Problem(
            480, 720, (Task("long", 90),), (Block(540, 570, "call"),)
        )
        result = self.assertSolved(problem)
        self.assertEqual(result.plan[0]["start"], 570)

    def test_infeasible_cases(self):
        cases = {
            "too much work": Problem(480, 600, (Task("a", 90), Task("b", 60))),
            "no gap big enough": Problem(
                480, 720, (Task("a", 90),),
                (Block(540, 570), Block(630, 660), Block(690, 720)),
            ),
            "impossible deadline": Problem(480, 720, (Task("a", 90, deadline=540),)),
            "dependency chain misses deadline": Problem(480, 720, (
                Task("a", 60), Task("b", 60, deadline=570, depends_on=("a",)),
            )),
            "earliest start too late": Problem(
                480, 720, (Task("a", 60, earliest_start=690),)
            ),
        }
        for name, problem in cases.items():
            with self.subTest(name):
                result = solve(problem)
                self.assertEqual(result.status, "infeasible")
                self.assertIsNone(result.plan)

    def test_node_limit_is_reported(self):
        problem, _ = generate_with_solution(
            1, n_tasks=9, n_blocks=2, utilization=0.9, dep_prob=0.0
        )
        result = solve(problem, max_nodes=2)
        self.assertEqual(result.status, "limit")


class RepairTest(unittest.TestCase):
    def test_unchanged_problem_needs_no_moves(self):
        problem, plan = generate_with_solution(5, n_tasks=6)
        result = repair(problem, plan)
        self.assertEqual(result.moved, 0)
        self.assertEqual(result.plan, plan)

    def test_only_conflicting_task_moves(self):
        problem = Problem(480, 780, (
            Task("a", 60), Task("b", 60), Task("c", 60),
        ))
        old = [entry("a", 480, 540), entry("b", 540, 600), entry("c", 600, 660)]
        changed = Problem(480, 780, problem.tasks, (Block(540, 600, "call"),))
        result = repair(changed, old)
        self.assertEqual(result.status, "solved")
        self.assertEqual(result.moved, 1)
        self.assertTrue(check_plan(changed, result.plan).valid)
        self.assertEqual(moved_tasks(old, result.plan), ["b"])

    def test_beats_resolving_from_scratch(self):
        # Old plan runs d, c, b, a. A new call blocks part of the last slot.
        # Repair keeps d, c, b where they are and moves only 'a'; solving
        # from scratch ignores the old plan and reshuffles tasks.
        problem = Problem(480, 900, (
            Task("a", 60), Task("b", 60), Task("c", 60), Task("d", 60),
        ))
        old = [entry("d", 480, 540), entry("c", 540, 600),
               entry("b", 600, 660), entry("a", 660, 720)]
        changed = Problem(480, 900, problem.tasks, (Block(690, 720, "call"),))
        repaired = repair(changed, old)
        scratch = solve(changed)
        self.assertEqual(repaired.moved, 1)   # only 'a' has to move
        self.assertGreaterEqual(
            len(moved_tasks(old, scratch.plan)), repaired.moved
        )

    def test_new_task_is_not_counted_as_moved(self):
        problem = Problem(480, 780, (Task("a", 60), Task("new", 60)))
        old = [entry("a", 480, 540)]
        result = repair(problem, old)
        self.assertEqual(result.moved, 0)
        self.assertEqual(
            {e["task_id"]: e["start"] for e in result.plan}["a"], 480
        )

    def test_removed_task_in_old_plan_is_ignored(self):
        problem = Problem(480, 780, (Task("a", 60),))
        old = [entry("a", 480, 540), entry("gone", 540, 600)]
        result = repair(problem, old)
        self.assertEqual(result.moved, 0)

    def test_infeasible_change(self):
        problem = Problem(480, 600, (Task("a", 120), Task("b", 60)))
        result = repair(problem, [entry("a", 480, 600)])
        self.assertEqual(result.status, "infeasible")

    def test_repair_is_never_worse_than_resolving(self):
        for seed in range(40):
            problem, plan = generate_with_solution(seed, n_tasks=7, n_blocks=2)
            change = make_change(problem, plan, seed)
            repaired = repair(change.problem, plan)
            scratch = solve(change.problem)
            self.assertTrue(repaired.solved, seed)
            self.assertTrue(check_plan(change.problem, repaired.plan).valid, seed)
            self.assertLessEqual(
                repaired.moved, len(moved_tasks(plan, scratch.plan)), seed
            )
            # the reported count matches an independent recount
            self.assertEqual(
                repaired.moved, len(moved_tasks(plan, repaired.plan)), seed
            )


class BruteForceAgreementTest(unittest.TestCase):
    """The solver must agree with exhaustive search on small problems."""

    SMALL = dict(n_tasks=3, n_blocks=1, utilization=0.6,
                 day_start=480, day_end=690)

    def test_feasibility_and_repair_optimality(self):
        checked = {"feasible": 0, "infeasible": 0}
        for seed in range(12):
            problem, plan = generate_with_solution(seed, **self.SMALL)
            for kind in CHANGE_KINDS:
                change = make_change(
                    problem, plan, seed * 10, kind=kind,
                    require_feasible=False, require_breaking=False,
                )
                changed = change.problem
                with self.subTest(seed=seed, kind=kind):
                    feasible, best_moved = brute_force(changed, plan)

                    solved = solve(changed)
                    self.assertEqual(solved.solved, feasible)
                    if solved.solved:
                        self.assertTrue(check_plan(changed, solved.plan).valid)

                    repaired = repair(changed, plan)
                    self.assertEqual(repaired.solved, feasible)
                    if feasible:
                        self.assertTrue(check_plan(changed, repaired.plan).valid)
                        self.assertEqual(repaired.moved, best_moved)
                        checked["feasible"] += 1
                    else:
                        checked["infeasible"] += 1

        # make sure the comparison really covered both kinds of outcome
        self.assertGreater(checked["feasible"], 15)
        self.assertGreater(checked["infeasible"], 2)

    def test_four_tasks(self):
        small = dict(n_tasks=4, n_blocks=1, utilization=0.6,
                     day_start=480, day_end=630)
        for seed in range(4):
            problem, plan = generate_with_solution(seed, **small)
            change = make_change(
                problem, plan, seed, require_feasible=False,
                require_breaking=False,
            )
            with self.subTest(seed=seed):
                feasible, best_moved = brute_force(change.problem, plan)
                repaired = repair(change.problem, plan)
                self.assertEqual(repaired.solved, feasible)
                if feasible:
                    self.assertEqual(repaired.moved, best_moved)


if __name__ == "__main__":
    unittest.main()
