"""LLM access layer: a small, testable wrapper around Gemini.

The experiments will make hundreds of LLM calls on a free tier with strict
limits, so this file exists to make those calls cheap, safe and repeatable.

Three layers, each doing one job:

    GeminiLLM   makes the real API call. Retries temporary failures (rate
                limits, overloaded servers) with growing waits, and can space
                requests out to stay under a requests-per-minute limit.
    CachedLLM   wraps any client and saves every successful response to disk.
                Asking the same question again is free, so a crashed or
                quota-limited experiment can simply be re-run and will pick
                up where it stopped, and results are reproducible.
    FakeLLM     returns scripted answers. Used in tests and for developing
                experiments offline without spending any quota.

All clients share one interface, `complete(prompt) -> LLMResult`, and never
raise on API failures: a failure comes back as `LLMResult(error=...)`, so an
experiment can record it as a result instead of crashing.

Quick check that your key and model work (makes one real call, then repeats
it to show the cache):

    python -m scheduler.llm
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Sequence, Union

RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}
RETRYABLE_WORDS = (
    "unavailable", "overloaded", "resource_exhausted", "rate limit",
    "timed out", "timeout", "try again",
)


class LLMConfigError(RuntimeError):
    """Raised when the key or model name is missing."""


@dataclass(frozen=True)
class LLMResult:
    """What came back from one call.

    text      the model's reply ("" if there was an error)
    error     description of the failure, or None on success
    cached    True if the reply came from the disk cache (no API request)
    attempts  API requests made, including retries (0 for cache hits)
    latency   seconds spent, including any waiting between retries
    """

    text: str = ""
    error: Optional[str] = None
    cached: bool = False
    attempts: int = 0
    latency: float = 0.0

    @property
    def ok(self) -> bool:
        return self.error is None


class LLMClient:
    """Interface shared by every backend."""

    def describe(self) -> dict:
        """Identify the model and settings; part of every cache key."""
        raise NotImplementedError

    def complete(self, prompt: str) -> LLMResult:
        raise NotImplementedError


# --------------------------------------------------------------------------
# Fake client (tests and offline development)
# --------------------------------------------------------------------------

class FakeLLM(LLMClient):
    """Scripted replies, so nothing here needs a network or an API key.

    responses  either a function prompt -> reply, or a list of replies that
               are returned one per call, in order.
    prompts    every prompt received, so tests can count and inspect calls.
    """

    def __init__(self, responses: Union[Callable, Sequence], name: str = "fake"):
        self._responses = responses
        self._next = 0
        self.name = name
        self.prompts = []

    def describe(self) -> dict:
        return {"model": self.name}

    def complete(self, prompt: str) -> LLMResult:
        self.prompts.append(prompt)
        if callable(self._responses):
            return LLMResult(text=self._responses(prompt), attempts=1)
        if self._next >= len(self._responses):
            return LLMResult(
                error="FakeLLM has no more scripted responses.", attempts=1
            )
        text = self._responses[self._next]
        self._next += 1
        return LLMResult(text=text, attempts=1)


# --------------------------------------------------------------------------
# Gemini client
# --------------------------------------------------------------------------

def _is_retryable(error: Exception) -> bool:
    """Temporary problems (rate limit, overloaded server) are worth retrying."""
    if getattr(error, "code", None) in RETRYABLE_STATUS_CODES:
        return True
    message = str(error).lower()
    return any(word in message for word in RETRYABLE_WORDS)


class GeminiLLM(LLMClient):
    """Real Gemini calls with retries and optional request spacing.

    model                 model name, e.g. "gemini-3.5-flash-lite"
    api_key               your key (not needed if `client` is given)
    generation_config     settings sent with every request. The default,
                          {"temperature": 0}, makes answers as repeatable as
                          possible. Pass {} to send no settings at all.
    max_retries           extra attempts after a temporary failure
    retry_base_seconds    first wait; it doubles each retry, up to the cap
    requests_per_minute   if set, calls are spaced to stay under this rate
    client                an already-built SDK client (used by the tests)
    sleep, clock          replaceable so tests never really wait
    """

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        generation_config: Optional[dict] = None,
        max_retries: int = 4,
        retry_base_seconds: float = 5.0,
        max_retry_seconds: float = 60.0,
        requests_per_minute: Optional[float] = None,
        client=None,
        sleep: Callable = time.sleep,
        clock: Callable = time.monotonic,
    ):
        if client is None:
            if not api_key:
                raise LLMConfigError("An api_key is required.")
            from google import genai  # imported here so tests do not need it
            client = genai.Client(api_key=api_key)

        self.model = model
        self.generation_config = (
            {"temperature": 0} if generation_config is None
            else dict(generation_config)
        )
        self.max_retries = max_retries
        self.retry_base_seconds = retry_base_seconds
        self.max_retry_seconds = max_retry_seconds
        self._min_interval = 60.0 / requests_per_minute if requests_per_minute else 0.0
        self._client = client
        self._sleep = sleep
        self._clock = clock
        self._last_request = None

    def describe(self) -> dict:
        return {"model": self.model, "generation_config": self.generation_config}

    def complete(self, prompt: str) -> LLMResult:
        start = self._clock()
        attempts = 0

        while True:
            self._wait_for_turn()
            attempts += 1
            try:
                text = self._generate(prompt)
            except Exception as error:  # the SDK raises many error types
                if attempts <= self.max_retries and _is_retryable(error):
                    self._sleep(self._backoff(attempts))
                    continue
                return LLMResult(
                    error=f"{type(error).__name__}: {error}",
                    attempts=attempts,
                    latency=self._clock() - start,
                )

            if not text or not text.strip():
                return LLMResult(
                    error="The model returned an empty response.",
                    attempts=attempts,
                    latency=self._clock() - start,
                )
            return LLMResult(
                text=text, attempts=attempts, latency=self._clock() - start
            )

    def _generate(self, prompt: str) -> str:
        """The only line that touches the Gemini SDK."""
        arguments = {"model": self.model, "input": prompt}
        if self.generation_config:
            arguments["generation_config"] = self.generation_config
        response = self._client.interactions.create(**arguments)
        return response.output_text

    def _backoff(self, attempt: int) -> float:
        wait = self.retry_base_seconds * 2 ** (attempt - 1)
        return min(wait, self.max_retry_seconds)

    def _wait_for_turn(self) -> None:
        """Pause if the previous request was too recent."""
        if self._min_interval and self._last_request is not None:
            wait = self._min_interval - (self._clock() - self._last_request)
            if wait > 0:
                self._sleep(wait)
        self._last_request = self._clock()


# --------------------------------------------------------------------------
# Disk cache
# --------------------------------------------------------------------------

class DiskCache:
    """One small JSON file per (model, settings, prompt) combination.

    The file stores the prompt next to the reply, so you can open any file in
    `cache/` and see exactly what was asked and answered.
    """

    def __init__(self, directory="cache"):
        self.directory = Path(directory)

    @staticmethod
    def make_key(identity: dict, prompt: str) -> str:
        payload = json.dumps(
            {"client": identity, "prompt": prompt},
            sort_keys=True,
            ensure_ascii=False,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _path(self, key: str) -> Path:
        return self.directory / key[:2] / f"{key}.json"

    def get(self, key: str) -> Optional[str]:
        try:
            data = json.loads(self._path(key).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None  # missing or corrupt file counts as a miss
        text = data.get("text") if isinstance(data, dict) else None
        return text if isinstance(text, str) else None

    def put(self, key: str, text: str, identity: dict, prompt: str) -> None:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        record = {"client": identity, "prompt": prompt, "text": text}
        temporary = path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        os.replace(temporary, path)  # atomic: no half-written cache files

    def __len__(self) -> int:
        return sum(1 for _ in self.directory.glob("*/*.json"))


class CachedLLM(LLMClient):
    """Adds the disk cache to any client. Only successes are cached.

    stats["api_calls"] counts real requests (retries included), which is what
    your free-tier quota counts; stats["cache_hits"] counts calls that cost
    nothing.
    """

    def __init__(self, inner: LLMClient, cache: DiskCache):
        self.inner = inner
        self.cache = cache
        self.stats = {"api_calls": 0, "cache_hits": 0, "errors": 0}

    def describe(self) -> dict:
        return self.inner.describe()

    def complete(self, prompt: str) -> LLMResult:
        identity = self.inner.describe()
        key = DiskCache.make_key(identity, prompt)

        cached_text = self.cache.get(key)
        if cached_text is not None:
            self.stats["cache_hits"] += 1
            return LLMResult(text=cached_text, cached=True)

        result = self.inner.complete(prompt)
        self.stats["api_calls"] += result.attempts
        if result.ok:
            self.cache.put(key, result.text, identity, prompt)
        else:
            self.stats["errors"] += 1
        return result


# --------------------------------------------------------------------------
# Convenience: build the normal client from your .env file
# --------------------------------------------------------------------------

def make_llm_from_env(cache_dir="cache", **gemini_options) -> CachedLLM:
    """Gemini + disk cache, configured from GEMINI_API_KEY / GEMINI_MODEL."""
    from dotenv import load_dotenv

    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    model = os.getenv("GEMINI_MODEL")
    if not api_key:
        raise LLMConfigError("GEMINI_API_KEY is not set. Add it to your .env file.")
    if not model:
        raise LLMConfigError("GEMINI_MODEL is not set. Add it to your .env file.")

    gemini = GeminiLLM(model, api_key=api_key, **gemini_options)
    return CachedLLM(gemini, DiskCache(cache_dir))


def main() -> int:
    prompt = " ".join(sys.argv[1:]) or (
        "Reply with one short sentence confirming you can respond."
    )
    try:
        llm = make_llm_from_env()
    except LLMConfigError as error:
        print(f"Setup problem: {error}")
        return 1

    print(f"Model: {llm.describe()['model']}")
    failed = False
    for label in ("first call", "second call (same prompt)"):
        result = llm.complete(prompt)
        if result.error:
            failed = True
            print(f"[{label}] ERROR after {result.attempts} request(s): "
                  f"{result.error}")
            continue
        source = (
            "from cache" if result.cached
            else f"from the API, {result.attempts} request(s)"
        )
        print(f"[{label}] {source}, {result.latency:.2f}s")
        print(f"    {result.text.strip()[:300]}")
    print("Stats:", llm.stats)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())