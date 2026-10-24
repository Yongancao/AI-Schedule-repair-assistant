import unittest

from scheduler.problem import (
    Block,
    Problem,
    ProblemError,
    Task,
    fmt_time,
    format_plan,
    format_problem,
    moved_tasks,
    parse_time,
    sort_plan,
    total_shift,
)


def sample_problem() -> Problem:
    return Problem(
        day_start=480,
        day_end=1020,
        blocks=(Block(720, 780, "meeting"),),
        tasks=(
            Task("write", 120, deadline=720),
            Task("review", 60, depends_on=("write",)),
            Task("gym", 60, earliest_start=840),
        ),
    )


class TimeHelpersTest(unittest.TestCase):
    def test_round_trip(self):
        for minutes in (0, 5, 60, 780, 1439, 1440):
            self.assertEqual(parse_time(fmt_time(minutes)), minutes)

    def test_format(self):
        self.assertEqual(fmt_time(780), "13:00")
        self.assertEqual(fmt_time(1440), "24:00")

    def test_parse_rejects_bad_text(self):
        for bad in ("25:00", "12:60", "noon", "12", "24:30", "-1:00"):
            with self.subTest(bad=bad):
                with self.assertRaises(ProblemError):
                    parse_time(bad)


class ProblemValidationTest(unittest.TestCase):
    def assertInvalid(self, **overrides):
        base = dict(day_start=480, day_end=1020, tasks=(Task("a", 30),))
        base.update(overrides)
        with self.assertRaises(ProblemError):
            Problem(**base)

    def test_valid_problem_builds(self):
        self.assertEqual(len(sample_problem().tasks), 3)

    def test_lists_are_converted_to_tuples(self):
        problem = Problem(480, 1020, [Task("a", 30, depends_on=["b"]), Task("b", 30)])
        self.assertIsInstance(problem.tasks, tuple)
        self.assertEqual(problem.tasks[0].depends_on, ("b",))

    def test_bad_window(self):
        self.assertInvalid(day_start=600, day_end=600)
        self.assertInvalid(day_start=-5)
        self.assertInvalid(day_end=1500)

    def test_duplicate_ids(self):
        self.assertInvalid(tasks=(Task("a", 30), Task("a", 45)))

    def test_bad_durations(self):
        self.assertInvalid(tasks=(Task("a", 0),))
        self.assertInvalid(tasks=(Task("a", -5),))
        self.assertInvalid(tasks=(Task("a", 30.5),))
        self.assertInvalid(tasks=(Task("a", True),))

    def test_bad_deadline_and_earliest_start(self):
        self.assertInvalid(tasks=(Task("a", 30, deadline=0),))
        self.assertInvalid(tasks=(Task("a", 30, deadline=2000),))
        self.assertInvalid(tasks=(Task("a", 30, earliest_start=-1),))

    def test_dependency_errors(self):
        self.assertInvalid(tasks=(Task("a", 30, depends_on=("zzz",)),))
        self.assertInvalid(tasks=(Task("a", 30, depends_on=("a",)),))

    def test_dependency_cycle(self):
        self.assertInvalid(tasks=(
            Task("a", 30, depends_on=("b",)),
            Task("b", 30, depends_on=("c",)),
            Task("c", 30, depends_on=("a",)),
        ))

    def test_bad_block(self):
        self.assertInvalid(blocks=(Block(700, 700),))
        self.assertInvalid(blocks=(Block(700, 2000),))


class FreeIntervalsTest(unittest.TestCase):
    def test_no_blocks(self):
        problem = Problem(480, 600, (Task("a", 30),))
        self.assertEqual(problem.free_intervals(), [(480, 600)])

    def test_blocks_split_the_day(self):
        self.assertEqual(
            sample_problem().free_intervals(), [(480, 720), (780, 1020)]
        )

    def test_overlapping_and_outside_blocks(self):
        problem = Problem(
            480, 1020, (),
            (Block(600, 700), Block(650, 800), Block(0, 100), Block(1100, 1200)),
        )
        self.assertEqual(problem.free_intervals(), [(480, 600), (800, 1020)])

    def test_block_touching_window_edges(self):
        problem = Problem(480, 600, (), (Block(480, 500), Block(580, 600)))
        self.assertEqual(problem.free_intervals(), [(500, 580)])

    def test_everything_blocked(self):
        problem = Problem(480, 600, (), (Block(400, 700),))
        self.assertEqual(problem.free_intervals(), [])
        self.assertEqual(problem.free_minutes(), 0)


class JsonRoundTripTest(unittest.TestCase):
    def test_round_trip(self):
        problem = sample_problem()
        self.assertEqual(Problem.from_dict(problem.to_dict()), problem)

    def test_optional_fields_are_omitted(self):
        data = Problem(480, 600, (Task("a", 30),)).to_dict()
        self.assertEqual(data["tasks"], [{"id": "a", "duration": 30}])
        self.assertEqual(data["blocks"], [])

    def test_missing_blocks_key_is_allowed(self):
        data = {"day_start": 480, "day_end": 600,
                "tasks": [{"id": "a", "duration": 30}]}
        self.assertEqual(len(Problem.from_dict(data).tasks), 1)

    def test_malformed_input_raises_problem_error(self):
        bad_inputs = [
            "not a dict",
            {},
            {"day_start": 480, "day_end": 600},
            {"day_start": 480, "day_end": 600, "tasks": [{"id": "a"}]},
            {"day_start": 480, "day_end": 600, "tasks": "oops"},
            {"day_start": "8am", "day_end": 600,
             "tasks": [{"id": "a", "duration": 30}]},
        ]
        for bad in bad_inputs:
            with self.subTest(bad=bad):
                with self.assertRaises(ProblemError):
                    Problem.from_dict(bad)


class PlanHelpersTest(unittest.TestCase):
    OLD = [
        {"task_id": "a", "start": 480, "end": 540},
        {"task_id": "b", "start": 540, "end": 600},
        {"task_id": "c", "start": 600, "end": 660},
    ]

    def test_moved_tasks_and_shift(self):
        new = [
            {"task_id": "a", "start": 480, "end": 540},   # unchanged
            {"task_id": "b", "start": 600, "end": 660},   # +60
            {"task_id": "c", "start": 570, "end": 630},   # -30
            {"task_id": "d", "start": 700, "end": 730},   # new task
        ]
        self.assertEqual(moved_tasks(self.OLD, new), ["b", "c"])
        self.assertEqual(total_shift(self.OLD, new), 90)

    def test_identical_plans_have_no_moves(self):
        self.assertEqual(moved_tasks(self.OLD, self.OLD), [])
        self.assertEqual(total_shift(self.OLD, self.OLD), 0)

    def test_sort_and_format(self):
        shuffled = [self.OLD[2], self.OLD[0], self.OLD[1]]
        self.assertEqual(sort_plan(shuffled), self.OLD)
        self.assertEqual(format_plan(shuffled).splitlines()[0], "08:00-09:00  a")

    def test_format_problem_mentions_everything(self):
        text = format_problem(sample_problem())
        for expected in ("08:00-17:00", "meeting", "write", "after write",
                         "not before 14:00", "done by 12:00"):
            self.assertIn(expected, text)


if __name__ == "__main__":
    unittest.main()
