import unittest

from scheduler.generator import generate_with_solution
from scheduler.parsing import (
    extract_json,
    format_plan_json,
    parse_clock,
    parse_reply,
)
from scheduler.problem import sort_plan
from scheduler.solver import solve

GOOD = '[{"task_id": "a", "start": "09:00", "end": "10:00"}]'
GOOD_PLAN = [{"task_id": "a", "start": 540, "end": 600}]


class ParseClockTest(unittest.TestCase):
    def test_valid_times(self):
        cases = {"09:00": 540, "9:00": 540, "00:00": 0, "13:05": 785,
                 " 13:05 ": 785, "24:00": 1440, "23:59": 1439}
        for text, expected in cases.items():
            with self.subTest(text=text):
                self.assertEqual(parse_clock(text), expected)

    def test_invalid_times(self):
        bad = ["25:00", "12:60", "24:30", "1:00 PM", "noon", "9", "09:0",
               "-1:00", "", "9:00:00", 900, None, 9.5, True]
        for value in bad:
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    parse_clock(value)


class ExtractJsonTest(unittest.TestCase):
    def test_plain_json(self):
        self.assertEqual(extract_json(GOOD)[0]["task_id"], "a")

    def test_code_fence(self):
        reply = f"Here you go:\n```json\n{GOOD}\n```\nHope that helps!"
        self.assertEqual(extract_json(reply)[0]["task_id"], "a")

    def test_fence_without_language(self):
        self.assertEqual(extract_json(f"```\n{GOOD}\n```")[0]["task_id"], "a")

    def test_text_around_unfenced_json(self):
        reply = f"Sure! The schedule is {GOOD} Let me know if you need more."
        self.assertEqual(extract_json(reply)[0]["task_id"], "a")

    def test_prefers_the_plan_over_stray_json(self):
        reply = f"See note [1] below. {GOOD}"
        self.assertEqual(extract_json(reply)[0]["task_id"], "a")

    def test_object_wrapper_is_returned_as_is(self):
        value = extract_json(f'{{"schedule": {GOOD}}}')
        self.assertIsInstance(value, dict)

    def test_unreadable_replies_raise(self):
        bad = ["", "   ", None, 42, "I could not do that.",
               '[{"task_id": "a", "start": "09:00"']   # cut off mid-way
        for reply in bad:
            with self.subTest(reply=reply):
                with self.assertRaises(ValueError):
                    extract_json(reply)


class ParseReplyTest(unittest.TestCase):
    def test_valid_reply(self):
        result = parse_reply(GOOD)
        self.assertTrue(result.ok)
        self.assertEqual(result.plan, GOOD_PLAN)
        self.assertEqual(result.errors, ())

    def test_extra_keys_are_ignored(self):
        reply = '[{"task_id": "a", "start": "09:00", "end": "10:00", "note": "x"}]'
        self.assertEqual(parse_reply(reply).plan, GOOD_PLAN)

    def test_wrapped_in_object(self):
        for key in ("plan", "schedule"):
            with self.subTest(key=key):
                self.assertEqual(
                    parse_reply(f'{{"{key}": {GOOD}}}').plan, GOOD_PLAN
                )

    def test_fenced_reply(self):
        self.assertEqual(parse_reply(f"```json\n{GOOD}\n```").plan, GOOD_PLAN)

    def test_empty_list_is_a_valid_empty_plan(self):
        result = parse_reply("[]")
        self.assertTrue(result.ok)
        self.assertEqual(result.plan, [])

    def test_unreadable_reply(self):
        result = parse_reply("Sorry, I can't.")
        self.assertFalse(result.ok)
        self.assertIsNone(result.plan)
        self.assertEqual(len(result.errors), 1)

    def test_wrong_top_level_shape(self):
        for reply in ('{"hello": "world"}', '"just text"', "42"):
            with self.subTest(reply=reply):
                result = parse_reply(reply)
                self.assertFalse(result.ok)
                self.assertTrue(result.errors)

    def test_entry_problems_are_all_reported(self):
        reply = (
            '[{"task_id": "a", "start": "9:00 AM", "end": "10:00"},'
            ' {"task_id": "b", "start": "09:00"},'
            ' {"start": "09:00", "end": "10:00"},'
            ' "oops",'
            ' {"task_id": "c", "start": 540, "end": "10:00"}]'
        )
        result = parse_reply(reply)
        self.assertFalse(result.ok)
        text = "\n".join(result.errors)
        self.assertEqual(len(result.errors), 5)
        for expected in ("Entry 1", "Entry 2", "Entry 3", "Entry 4", "Entry 5"):
            self.assertIn(expected, text)
        self.assertIn("missing 'end'", text)

    def test_one_bad_entry_makes_the_whole_reply_unreadable(self):
        reply = ('[{"task_id": "a", "start": "09:00", "end": "10:00"},'
                 ' {"task_id": "b", "start": "noon", "end": "13:00"}]')
        self.assertIsNone(parse_reply(reply).plan)

    def test_order_is_preserved(self):
        reply = ('[{"task_id": "b", "start": "11:00", "end": "12:00"},'
                 ' {"task_id": "a", "start": "09:00", "end": "10:00"}]')
        self.assertEqual([e["task_id"] for e in parse_reply(reply).plan], ["b", "a"])

    def test_non_text_reply(self):
        self.assertFalse(parse_reply(None).ok)


class FormatPlanJsonTest(unittest.TestCase):
    def test_format(self):
        self.assertEqual(
            format_plan_json(GOOD_PLAN),
            '[\n  {"task_id": "a", "start": "09:00", "end": "10:00"}\n]',
        )

    def test_empty(self):
        self.assertEqual(format_plan_json([]), "[]")

    def test_sorted_by_start(self):
        plan = [{"task_id": "b", "start": 600, "end": 660},
                {"task_id": "a", "start": 540, "end": 600}]
        self.assertLess(format_plan_json(plan).index('"a"'),
                        format_plan_json(plan).index('"b"'))

    def test_round_trip_through_the_parser(self):
        for seed in range(30):
            problem, _ = generate_with_solution(seed, n_tasks=7, n_blocks=2)
            plan = solve(problem).plan
            parsed = parse_reply(format_plan_json(plan))
            self.assertTrue(parsed.ok, seed)
            self.assertEqual(parsed.plan, sort_plan(plan), seed)


if __name__ == "__main__":
    unittest.main()