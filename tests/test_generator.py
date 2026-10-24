import unittest

from scheduler.checker import check_plan
from scheduler.generator import (
    CHANGE_KINDS,
    GRID,
    PRESETS,
    GenerationError,
    generate_preset,
    generate_problem,
    generate_with_solution,
    make_change,
)
from scheduler.solver import solve


class GenerateProblemTest(unittest.TestCase):
    def test_planted_plan_is_always_valid(self):
        for name in PRESETS:
            for seed in range(60):
                with self.subTest(preset=name, seed=seed):
                    problem, plan = generate_preset(name, seed)
                    self.assertTrue(check_plan(problem, plan).valid)

    def test_generated_problems_are_solvable(self):
        for name in PRESETS:
            for seed in range(30):
                with self.subTest(preset=name, seed=seed):
                    problem = generate_preset(name, seed)[0]
                    result = solve(problem)
                    self.assertTrue(result.solved)
                    self.assertTrue(check_plan(problem, result.plan).valid)

    def test_same_seed_gives_same_problem(self):
        self.assertEqual(generate_problem(7, n_tasks=6), generate_problem(7, n_tasks=6))

    def test_different_seeds_give_different_problems(self):
        problems = {generate_problem(seed, n_tasks=6) for seed in range(10)}
        self.assertGreater(len(problems), 5)

    def test_knobs_are_respected(self):
        problem = generate_problem(3, n_tasks=8, n_blocks=3, utilization=0.6)
        self.assertEqual(len(problem.tasks), 8)
        self.assertEqual(len(problem.blocks), 3)
        used = problem.total_duration() / problem.free_minutes()
        self.assertLessEqual(used, 1.0)

    def test_everything_is_on_the_time_grid(self):
        problem, plan = generate_with_solution(11, n_tasks=7, n_blocks=2)
        for task in problem.tasks:
            self.assertEqual(task.duration % GRID, 0)
        for item in plan:
            self.assertEqual(item["start"] % GRID, 0)

    def test_task_order_does_not_leak_the_planted_order(self):
        # The task list is shuffled, so it should usually differ from plan order.
        differing = 0
        for seed in range(20):
            problem, plan = generate_with_solution(seed, n_tasks=6)
            if problem.task_ids() != [e["task_id"] for e in plan]:
                differing += 1
        self.assertGreater(differing, 10)

    def test_prerequisites_are_acyclic_and_known(self):
        # Problem validation enforces this; generation must never trip it.
        for seed in range(50):
            generate_problem(seed, n_tasks=9, dep_prob=0.9)

    def test_impossible_settings_raise(self):
        with self.assertRaises(GenerationError):
            generate_with_solution(0, n_tasks=30, utilization=1.0,
                                   n_blocks=4, max_attempts=20)


class MakeChangeTest(unittest.TestCase):
    def setUp(self):
        self.problem, self.plan = generate_preset("medium", 4)

    def test_every_kind_breaks_the_plan_and_stays_feasible(self):
        for kind in CHANGE_KINDS:
            for seed in range(10):
                with self.subTest(kind=kind, seed=seed):
                    change = make_change(self.problem, self.plan, seed, kind=kind)
                    self.assertEqual(change.kind, kind)
                    self.assertTrue(change.description)
                    self.assertFalse(
                        check_plan(change.problem, self.plan).valid
                    )
                    self.assertTrue(solve(change.problem).solved)

    def test_random_kind_is_reproducible(self):
        first = make_change(self.problem, self.plan, 99)
        second = make_change(self.problem, self.plan, 99)
        self.assertEqual(first, second)

    def test_change_does_not_modify_the_original(self):
        before = self.problem
        make_change(self.problem, self.plan, 1)
        self.assertEqual(self.problem, before)

    def test_unknown_kind(self):
        with self.assertRaises(ValueError):
            make_change(self.problem, self.plan, 0, kind="explode")

    def test_changes_across_many_problems(self):
        for seed in range(30):
            problem, plan = generate_preset("hard", seed)
            change = make_change(problem, plan, seed)
            self.assertFalse(check_plan(change.problem, plan).valid)
            self.assertTrue(solve(change.problem).solved)


if __name__ == "__main__":
    unittest.main()
