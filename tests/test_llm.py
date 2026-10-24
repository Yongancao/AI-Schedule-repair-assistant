import tempfile
import unittest
from types import SimpleNamespace

from scheduler.llm import (
    CachedLLM,
    DiskCache,
    FakeLLM,
    GeminiLLM,
    LLMConfigError,
    LLMResult,
)


class FakeTime:
    """A clock that only moves when someone sleeps, so tests never wait."""

    def __init__(self):
        self.now = 1000.0
        self.sleeps = []

    def clock(self):
        return self.now

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds


class FakeAPIError(Exception):
    """Looks like the SDK's errors: has an HTTP status in `.code`."""

    def __init__(self, code, message="boom"):
        super().__init__(message)
        self.code = code


class FakeSDKClient:
    """Stands in for genai.Client. Each step is an Exception or a text reply."""

    def __init__(self, steps):
        self.steps = list(steps)
        self.requests = []
        self.interactions = SimpleNamespace(create=self._create)

    def _create(self, **kwargs):
        self.requests.append(kwargs)
        step = self.steps.pop(0)
        if isinstance(step, Exception):
            raise step
        return SimpleNamespace(output_text=step)


def make_gemini(steps, **options):
    timer = FakeTime()
    sdk = FakeSDKClient(steps)
    llm = GeminiLLM(
        "test-model", client=sdk, sleep=timer.sleep, clock=timer.clock, **options
    )
    return llm, sdk, timer


class FakeLLMTest(unittest.TestCase):
    def test_scripted_list_in_order(self):
        llm = FakeLLM(["one", "two"])
        self.assertEqual(llm.complete("a").text, "one")
        self.assertEqual(llm.complete("b").text, "two")
        self.assertEqual(llm.prompts, ["a", "b"])

    def test_running_out_of_responses_is_an_error_result(self):
        llm = FakeLLM(["only"])
        llm.complete("a")
        result = llm.complete("b")
        self.assertFalse(result.ok)
        self.assertIn("no more", result.error)

    def test_function_responder(self):
        llm = FakeLLM(lambda prompt: prompt.upper())
        self.assertEqual(llm.complete("hi").text, "HI")


class GeminiLLMTest(unittest.TestCase):
    def test_success_sends_model_prompt_and_default_temperature(self):
        llm, sdk, _ = make_gemini(["hello"])
        result = llm.complete("question")
        self.assertTrue(result.ok)
        self.assertEqual(result.text, "hello")
        self.assertEqual(result.attempts, 1)
        self.assertEqual(sdk.requests, [{
            "model": "test-model",
            "input": "question",
            "generation_config": {"temperature": 0},
        }])

    def test_empty_generation_config_sends_no_settings(self):
        llm, sdk, _ = make_gemini(["hello"], generation_config={})
        llm.complete("question")
        self.assertNotIn("generation_config", sdk.requests[0])

    def test_custom_generation_config(self):
        llm, sdk, _ = make_gemini(
            ["hello"], generation_config={"temperature": 0.5}
        )
        llm.complete("question")
        self.assertEqual(sdk.requests[0]["generation_config"], {"temperature": 0.5})

    def test_retries_temporary_errors_with_growing_waits(self):
        llm, sdk, timer = make_gemini(
            [FakeAPIError(503), FakeAPIError(429), "finally"],
            retry_base_seconds=5.0,
        )
        result = llm.complete("q")
        self.assertTrue(result.ok)
        self.assertEqual(result.text, "finally")
        self.assertEqual(result.attempts, 3)
        self.assertEqual(timer.sleeps, [5.0, 10.0])
        self.assertEqual(len(sdk.requests), 3)

    def test_gives_up_after_max_retries(self):
        llm, sdk, timer = make_gemini(
            [FakeAPIError(503)] * 10, max_retries=2, retry_base_seconds=1.0
        )
        result = llm.complete("q")
        self.assertFalse(result.ok)
        self.assertEqual(result.attempts, 3)          # 1 try + 2 retries
        self.assertEqual(timer.sleeps, [1.0, 2.0])
        self.assertIn("FakeAPIError", result.error)

    def test_backoff_is_capped(self):
        llm, _, timer = make_gemini(
            [FakeAPIError(503)] * 10,
            max_retries=5, retry_base_seconds=10.0, max_retry_seconds=25.0,
        )
        llm.complete("q")
        self.assertEqual(timer.sleeps, [10.0, 20.0, 25.0, 25.0, 25.0])

    def test_permanent_errors_are_not_retried(self):
        llm, sdk, timer = make_gemini([FakeAPIError(400, "bad request")])
        result = llm.complete("q")
        self.assertFalse(result.ok)
        self.assertEqual(result.attempts, 1)
        self.assertEqual(timer.sleeps, [])
        self.assertIn("bad request", result.error)

    def test_errors_recognised_by_message_text(self):
        llm, _, _ = make_gemini(
            [RuntimeError("503 UNAVAILABLE: model is overloaded"), "ok"]
        )
        self.assertEqual(llm.complete("q").attempts, 2)

    def test_empty_reply_is_an_error(self):
        for empty in ("", "   \n", None):
            with self.subTest(empty=empty):
                llm, _, _ = make_gemini([empty])
                result = llm.complete("q")
                self.assertFalse(result.ok)
                self.assertIn("empty", result.error)

    def test_requests_are_spaced_to_respect_the_rate_limit(self):
        llm, _, timer = make_gemini(["a", "b", "c"], requests_per_minute=30)
        for _ in range(3):
            llm.complete("q")
        # 30 requests/minute = one every 2 seconds. No wait before the first.
        self.assertEqual(timer.sleeps, [2.0, 2.0])

    def test_no_spacing_when_no_limit_is_set(self):
        llm, _, timer = make_gemini(["a", "b"])
        llm.complete("q")
        llm.complete("q")
        self.assertEqual(timer.sleeps, [])

    def test_latency_includes_waiting(self):
        llm, _, _ = make_gemini([FakeAPIError(503), "ok"], retry_base_seconds=5.0)
        self.assertEqual(llm.complete("q").latency, 5.0)

    def test_missing_api_key(self):
        with self.assertRaises(LLMConfigError):
            GeminiLLM("test-model")

    def test_describe_identifies_model_and_settings(self):
        llm, _, _ = make_gemini(["x"])
        self.assertEqual(llm.describe(), {
            "model": "test-model", "generation_config": {"temperature": 0},
        })


class DiskCacheTest(unittest.TestCase):
    def test_round_trip_and_miss(self):
        with tempfile.TemporaryDirectory() as folder:
            cache = DiskCache(folder)
            key = DiskCache.make_key({"model": "m"}, "prompt")
            self.assertIsNone(cache.get(key))
            cache.put(key, "reply", {"model": "m"}, "prompt")
            self.assertEqual(cache.get(key), "reply")
            self.assertEqual(len(cache), 1)

    def test_unicode_survives(self):
        with tempfile.TemporaryDirectory() as folder:
            cache = DiskCache(folder)
            key = DiskCache.make_key({}, "café ☕")
            cache.put(key, "résumé – naïve", {}, "café ☕")
            self.assertEqual(cache.get(key), "résumé – naïve")

    def test_key_depends_on_prompt_model_and_settings(self):
        base = DiskCache.make_key({"model": "a", "generation_config": {}}, "p")
        self.assertEqual(base, DiskCache.make_key(
            {"generation_config": {}, "model": "a"}, "p"))   # order irrelevant
        self.assertNotEqual(base, DiskCache.make_key(
            {"model": "a", "generation_config": {}}, "p2"))
        self.assertNotEqual(base, DiskCache.make_key(
            {"model": "b", "generation_config": {}}, "p"))
        self.assertNotEqual(base, DiskCache.make_key(
            {"model": "a", "generation_config": {"temperature": 1}}, "p"))

    def test_corrupt_file_counts_as_a_miss(self):
        with tempfile.TemporaryDirectory() as folder:
            cache = DiskCache(folder)
            key = DiskCache.make_key({}, "p")
            cache.put(key, "reply", {}, "p")
            cache._path(key).write_text("{not json", encoding="utf-8")
            self.assertIsNone(cache.get(key))

    def test_file_without_text_counts_as_a_miss(self):
        with tempfile.TemporaryDirectory() as folder:
            cache = DiskCache(folder)
            key = DiskCache.make_key({}, "p")
            cache.put(key, "reply", {}, "p")
            cache._path(key).write_text('{"other": 1}', encoding="utf-8")
            self.assertIsNone(cache.get(key))


class CachedLLMTest(unittest.TestCase):
    def test_second_identical_call_is_free(self):
        with tempfile.TemporaryDirectory() as folder:
            inner = FakeLLM(lambda prompt: "answer to " + prompt)
            llm = CachedLLM(inner, DiskCache(folder))

            first = llm.complete("q")
            second = llm.complete("q")

            self.assertFalse(first.cached)
            self.assertTrue(second.cached)
            self.assertEqual(second.text, "answer to q")
            self.assertEqual(len(inner.prompts), 1)
            self.assertEqual(llm.stats,
                             {"api_calls": 1, "cache_hits": 1, "errors": 0})

    def test_different_prompts_are_cached_separately(self):
        with tempfile.TemporaryDirectory() as folder:
            inner = FakeLLM(lambda prompt: prompt)
            llm = CachedLLM(inner, DiskCache(folder))
            llm.complete("a")
            llm.complete("b")
            self.assertEqual(len(inner.prompts), 2)

    def test_cache_survives_a_restart(self):
        with tempfile.TemporaryDirectory() as folder:
            CachedLLM(FakeLLM(["saved"]), DiskCache(folder)).complete("q")

            new_inner = FakeLLM([])        # would error if it were asked
            result = CachedLLM(new_inner, DiskCache(folder)).complete("q")
            self.assertEqual(result.text, "saved")
            self.assertTrue(result.cached)
            self.assertEqual(new_inner.prompts, [])

    def test_errors_are_not_cached(self):
        with tempfile.TemporaryDirectory() as folder:
            inner = FakeLLM([])            # every call is an error result
            llm = CachedLLM(inner, DiskCache(folder))
            self.assertFalse(llm.complete("q").ok)
            self.assertFalse(llm.complete("q").ok)
            self.assertEqual(len(inner.prompts), 2)
            self.assertEqual(llm.stats["errors"], 2)
            self.assertEqual(len(DiskCache(folder)), 0)

    def test_different_models_do_not_share_cache_entries(self):
        with tempfile.TemporaryDirectory() as folder:
            cache = DiskCache(folder)
            CachedLLM(FakeLLM(["from a"], name="a"), cache).complete("q")
            result = CachedLLM(FakeLLM(["from b"], name="b"), cache).complete("q")
            self.assertEqual(result.text, "from b")

    def test_stats_count_retries_as_requests(self):
        with tempfile.TemporaryDirectory() as folder:
            gemini, _, _ = make_gemini([FakeAPIError(503), "ok"])
            llm = CachedLLM(gemini, DiskCache(folder))
            llm.complete("q")
            self.assertEqual(llm.stats["api_calls"], 2)


class LLMResultTest(unittest.TestCase):
    def test_ok_flag(self):
        self.assertTrue(LLMResult(text="x").ok)
        self.assertFalse(LLMResult(error="bad").ok)


if __name__ == "__main__":
    unittest.main()