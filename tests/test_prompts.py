import unittest

from scheduler.generator import generate_with_solution
from scheduler.problem import Block, Problem, Task
from scheduler.prompts import (
    MAX_PREVIOUS_REPLY_CHARS,
    build_feedback_prompt,
    build_plan_prompt,
    problem_to_text,
)


def sample_problem() -> Problem:
    return Problem(
        day_start=480,
        day_end=1020,
        blocks=(Block(720, 780, "team_meeting"),),
        tasks=(
            Task("write", 120, deadline=720),
            Task("review", 60, depends_on=("write",)),
            Task("gym", 60, earliest_start=840, deadline=960),
        ),
    )


class ProblemToTextTest(unittest.TestCase):
    def test_everything_appears_in_hhmm(self):
        text = problem_to_text(sample_problem())
        for expected in (
            "08:00 to 17:00",
            "team_meeting: 12:00 to 13:00",
            "write: 120 minutes",
            "must finish by 12:00",
            "review: 60 minutes",
            "can only start after these have finished: write",
            "cannot start before 14:00",
            "must finish by 16:00",
        ):
            self.assertIn(expected, text)

    def test_raw_minute_numbers_are_not_shown(self):
        text = problem_to_text(sample_problem())
        for raw in ("480", "1020", "720", "780", "840", "960"):
            self.assertNotIn(raw, text)

    def test_no_fixed_events(self):
        text = problem_to_text(Problem(480, 600, (Task("a", 30),)))
        self.assertIn("Fixed events: none", text)

    def test_every_generated_task_is_listed(self):
        for seed in range(20):
            problem, _ = generate_with_solution(seed, n_tasks=8, n_blocks=2)
            text = problem_to_text(problem)
            for task in problem.tasks:
                self.assertIn(f"- {task.id}:", text)


class PlanPromptTest(unittest.TestCase):
    def test_contains_rules_format_and_problem(self):
        prompt = build_plan_prompt(sample_problem())
        for expected in ("Rules:", "exactly once", "24-hour HH:MM",
                         "JSON list", '"task_id"', "write: 120 minutes"):
            self.assertIn(expected, prompt)

    def test_is_deterministic(self):
        self.assertEqual(build_plan_prompt(sample_problem()),
                         build_plan_prompt(sample_problem()))

    def test_different_problems_give_different_prompts(self):
        other = Problem(480, 1020, (Task("something_else", 30),))
        self.assertNotEqual(build_plan_prompt(sample_problem()),
                            build_plan_prompt(other))


class FeedbackPromptTest(unittest.TestCase):
    PLAN = [{"task_id": "write", "start": 480, "end": 600}]
    MESSAGES = ["Required task 'gym' is missing from the plan.",
                "Task 'review' starts too early."]

    def test_contains_original_prompt_plan_and_messages(self):
        problem = sample_problem()
        prompt = build_feedback_prompt(problem, "raw reply", self.PLAN, self.MESSAGES)
        self.assertTrue(prompt.startswith(build_plan_prompt(problem)))
        self.assertIn("Your previous answer was:", prompt)
        self.assertIn('"start": "08:00"', prompt)          # cleaned-up plan
        self.assertNotIn("raw reply", prompt)
        for message in self.MESSAGES:
            self.assertIn(f"- {message}", prompt)
        self.assertIn("COMPLETE corrected schedule", prompt)

    def test_unreadable_reply_is_shown_raw(self):
        prompt = build_feedback_prompt(
            sample_problem(), "I think maybe 9am?", None, ["could not be read"]
        )
        self.assertIn("I think maybe 9am?", prompt)

    def test_very_long_raw_reply_is_shortened(self):
        long_reply = "x" * (MAX_PREVIOUS_REPLY_CHARS * 3)
        prompt = build_feedback_prompt(sample_problem(), long_reply, None, ["bad"])
        self.assertIn("(shortened)", prompt)
        self.assertLess(prompt.count("x"), MAX_PREVIOUS_REPLY_CHARS + 50)

    def test_only_the_latest_attempt_is_included(self):
        # the prompt is rebuilt from scratch each round, so it cannot grow
        problem = sample_problem()
        first = build_feedback_prompt(problem, "r", self.PLAN, ["problem one"])
        second = build_feedback_prompt(problem, "r", self.PLAN, ["problem two"])
        self.assertNotIn("problem one", second)
        self.assertEqual(len(first.splitlines()), len(second.splitlines()))


if __name__ == "__main__":
    unittest.main()