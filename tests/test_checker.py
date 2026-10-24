import unittest

from scheduler.checker import VIOLATION_TYPES, check_plan
from scheduler.problem import Block, Problem, Task


def make_problem() -> Problem:
    """08:00-17:00, a meeting 12:00-13:00, and three tasks."""
    return Problem(
        day_start=480,
        day_end=1020,
        blocks=(Block(720, 780, "meeting"),),
        tasks=(
            Task("write", 120, deadline=720),                    # done by 12:00
            Task("review", 60, depends_on=("write",)),
            Task("gym", 60, earliest_start=840),                 # not before 14:00
        ),
    )


def entry(task_id, start, end):
    return {"task_id": task_id, "start": start, "end": end}


GOOD_PLAN = [
    entry("write", 480, 600),
    entry("review", 600, 660),
    entry("gym", 840, 900),
]


class CheckerTest(unittest.TestCase):
    def setUp(self):
        self.problem = make_problem()

    def types(self, plan):
        return check_plan(self.problem, plan).types()

    def test_valid_plan(self):
        result = check_plan(self.problem, GOOD_PLAN)
        self.assertTrue(result.valid)
        self.assertEqual(result.violations, ())

    def test_tasks_may_touch_each_other_and_blocks(self):
        plan = [entry("write", 480, 600), entry("review", 600, 660),
                entry("gym", 780, 840)]
        # gym at 13:00 violates earliest_start, but touching the meeting
        # (ends exactly 13:00) must not count as a conflict.
        self.assertEqual(self.types(plan), ["before_earliest_start"])

    def test_missing_task(self):
        self.assertEqual(self.types(GOOD_PLAN[:2]), ["missing_task"])

    def test_unknown_task(self):
        self.assertEqual(
            self.types(GOOD_PLAN + [entry("nap", 900, 930)]), ["unknown_task"]
        )

    def test_duplicate_task(self):
        plan = GOOD_PLAN + [entry("gym", 900, 960)]
        self.assertIn("duplicate_task", self.types(plan))

    def test_bad_time_range_does_not_cascade(self):
        plan = [entry("write", 480, 600), entry("review", 660, 660),
                entry("gym", 840, 900)]
        self.assertEqual(self.types(plan), ["bad_time_range"])

    def test_negative_and_past_midnight_times(self):
        for bad in (entry("gym", -30, 30), entry("gym", 1400, 1500)):
            plan = GOOD_PLAN[:2] + [bad]
            self.assertIn("bad_time_range", self.types(plan))

    def test_wrong_duration(self):
        plan = [entry("write", 480, 600), entry("review", 600, 650),
                entry("gym", 840, 900)]
        self.assertEqual(self.types(plan), ["wrong_duration"])

    def test_outside_availability(self):
        plan = [entry("write", 480, 600), entry("review", 600, 660),
                entry("gym", 990, 1050)]
        self.assertEqual(self.types(plan), ["outside_availability"])

    def test_before_earliest_start(self):
        plan = [entry("write", 480, 600), entry("review", 600, 660),
                entry("gym", 780, 840)]
        self.assertEqual(self.types(plan), ["before_earliest_start"])

    def test_missed_deadline(self):
        plan = [entry("write", 780, 900), entry("review", 900, 960),
                entry("gym", 960, 1020)]
        self.assertEqual(self.types(plan), ["missed_deadline"])

    def test_overlap(self):
        plan = [entry("write", 480, 600), entry("review", 540, 600),
                entry("gym", 840, 900)]
        self.assertIn("overlap", self.types(plan))

    def test_blocked_conflict(self):
        plan = [entry("write", 600, 720), entry("review", 700, 760),
                entry("gym", 840, 900)]
        self.assertIn("blocked_conflict", self.types(plan))
        message = check_plan(self.problem, plan).messages()
        self.assertTrue(any("meeting" in m for m in message))

    def test_dependency_violation(self):
        plan = [entry("review", 480, 540), entry("write", 540, 660),
                entry("gym", 840, 900)]
        self.assertEqual(self.types(plan), ["dependency_violation"])

    def test_dependency_not_checked_when_prerequisite_missing(self):
        # 'write' is missing: only missing_task is reported, not a dependency error
        plan = [entry("review", 600, 660), entry("gym", 840, 900)]
        self.assertEqual(self.types(plan), ["missing_task"])

    def test_multiple_violations_are_all_reported(self):
        plan = [entry("write", 780, 900), entry("review", 780, 840),
                entry("gym", 700, 760)]
        found = set(self.types(plan))
        self.assertTrue({"missed_deadline", "overlap", "blocked_conflict",
                         "before_earliest_start",
                         "dependency_violation"} <= found)

    def test_counts_and_messages(self):
        plan = [entry("write", 480, 600), entry("review", 540, 600)]
        result = check_plan(self.problem, plan)
        self.assertEqual(result.counts()["missing_task"], 1)
        self.assertEqual(len(result.messages()), len(result.violations))
        self.assertTrue(all(isinstance(m, str) and m for m in result.messages()))

    def test_all_reported_types_are_known(self):
        plan = [entry("write", 780, 900), entry("review", 780, 840),
                entry("gym", 700, 760), entry("nap", 0, 10)]
        for violation_type in self.types(plan):
            self.assertIn(violation_type, VIOLATION_TYPES)


class StructureTest(unittest.TestCase):
    def setUp(self):
        self.problem = make_problem()

    def assertOnlyStructureErrors(self, plan):
        result = check_plan(self.problem, plan)
        self.assertFalse(result.valid)
        self.assertEqual(set(result.types()), {"invalid_structure"})

    def test_plan_must_be_a_list(self):
        for bad in ("nope", None, {"task_id": "gym"}, 42):
            with self.subTest(bad=bad):
                self.assertOnlyStructureErrors(bad)

    def test_entries_must_be_objects(self):
        self.assertOnlyStructureErrors(["write"])

    def test_missing_fields(self):
        self.assertOnlyStructureErrors([{"task_id": "gym", "start": 840}])
        self.assertOnlyStructureErrors([{"start": 840, "end": 900}])

    def test_wrong_types(self):
        self.assertOnlyStructureErrors([entry("gym", 840.0, 900)])
        self.assertOnlyStructureErrors([entry("gym", "14:00", "15:00")])
        self.assertOnlyStructureErrors([entry("gym", True, 900)])
        self.assertOnlyStructureErrors([entry(7, 840, 900)])
        self.assertOnlyStructureErrors([entry("", 840, 900)])

    def test_empty_plan_reports_every_missing_task(self):
        result = check_plan(self.problem, [])
        self.assertEqual(result.types(), ["missing_task"] * 3)


if __name__ == "__main__":
    unittest.main()
