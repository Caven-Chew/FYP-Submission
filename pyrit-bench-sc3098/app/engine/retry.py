"""Sending with retry/backoff, plus the consecutive-failure abort guard."""
from __future__ import annotations

import asyncio

from ..activity import active_benchmark, set_activity
from ..settings import RETRY_BASE_DELAY, RETRY_MAX_ATTEMPTS, RETRY_MAX_DELAY


class BenchmarkAborted(Exception):
    """Raised to stop a benchmark early once too many target calls fail in a row."""


#: Exception names that mean THIS MACHINE is broken, not the target. A missing
#: browser binary or a package data file that vanished says nothing whatsoever
#: about whether the endpoint is reachable, and telling the user to go check
#: their endpoint and credentials sends them hunting in the wrong place —
#: benchmark 57 aborted with "the target appears unreachable. Last error:
#: FileNotFoundError: .../patchright/driver/node", which was a deleted venv.
_LOCAL_ENVIRONMENT_ERRORS = (
    "FileNotFoundError", "ModuleNotFoundError", "ImportError",
    "PermissionError", "NotADirectoryError", "IsADirectoryError",
)


def looks_like_local_problem(error: str | None) -> bool:
    """True when the error names a fault on the machine running the benchmark."""
    if not error:
        return False
    head = error.strip()[:80]
    return any(name in head for name in _LOCAL_ENVIRONMENT_ERRORS)


class ConsecutiveFailureGuard:
    """Tracks consecutive external target failures across one benchmark run.

    Shared by every attack in the run (static, dynamic, Azure RedTeam) so a
    dead target is caught regardless of which attack happens to be running
    when it becomes clear the target isn't responding. `record(error)` resets
    the streak to 0 on success (falsy `error`) and raises `BenchmarkAborted`
    once `limit` failures have landed back to back, unless `stop_on_limit` is
    False. Benchmark runs use False: failures stay on their result rows and
    determine the final status, while remaining tests are still attempted.
    """

    def __init__(self, limit: int, *, stop_on_limit: bool = True):
        self.limit = limit
        self.stop_on_limit = stop_on_limit
        self.consecutive = 0
        self.last_error: str | None = None

    @property
    def last_error_is_local(self) -> bool:
        return looks_like_local_problem(self.last_error)

    def abort_message(self) -> str:
        """Why the run stopped, blaming whatever actually broke."""
        if self.last_error_is_local:
            return (f"{self.consecutive} calls in a row failed before reaching the "
                    f"target — this looks like a problem on this machine, not with "
                    f"the target. Last error: {self.last_error}")
        return (f"{self.consecutive} consecutive target calls failed — the target "
                f"appears unreachable. Last error: {self.last_error}")

    def record(self, error: str | None) -> None:
        if not error:
            self.consecutive = 0
            return
        self.consecutive += 1
        self.last_error = error
        if self.tripped:
            raise BenchmarkAborted(self.abort_message())

    @property
    def tripped(self) -> bool:
        return self.stop_on_limit and self.limit > 0 and self.consecutive >= self.limit


def retryable_error(exc: Exception) -> tuple[bool, float | None]:
    """Classify an exception: (should_retry, retry_after_seconds_or_None).

    Retries rate limits (429), transient 5xx, timeouts and connection errors, and
    honours an HTTP ``Retry-After`` header when the provider sends one. Does NOT
    retry auth / bad-request 4xx — those won't fix themselves.
    """
    try:
        import httpx
        if isinstance(exc, httpx.HTTPStatusError):
            code = exc.response.status_code
            if code == 429 or 500 <= code < 600:
                ra = exc.response.headers.get("retry-after")
                try:
                    return True, float(ra) if ra else None
                except (TypeError, ValueError):
                    return True, None
            return False, None
        if isinstance(exc, (httpx.TimeoutException, httpx.TransportError)):
            return True, None
    except Exception:  # noqa: BLE001 — httpx isn't the error's type
        pass
    # Fallback: match common rate-limit / transient signals in the message text
    msg = f"{type(exc).__name__}: {exc}".lower()
    signals = ("rate limit", "ratelimit", "429", "too many requests", "timeout",
               "timed out", "overloaded", "temporarily unavailable",
               "502", "503", "504", "service unavailable")
    return (any(s in msg for s in signals), None)


async def retry_async(make_coro, *, what: str = "LLM call"):
    """Await ``make_coro()`` with exponential backoff on rate-limit / transient
    errors. ``make_coro`` is a zero-arg factory returning a fresh coroutine each
    attempt. Non-retryable errors raise immediately; the last error is re-raised
    after ``LLM_RETRY_MAX_ATTEMPTS`` attempts.
    """
    import random
    for attempt in range(RETRY_MAX_ATTEMPTS):
        try:
            return await make_coro()
        except Exception as exc:  # noqa: BLE001
            retryable, retry_after = retryable_error(exc)
            if not retryable or attempt == RETRY_MAX_ATTEMPTS - 1:
                raise
            delay = retry_after if retry_after is not None else min(
                RETRY_MAX_DELAY, RETRY_BASE_DELAY * (2 ** attempt)
            )
            delay += random.uniform(0, delay * 0.25)  # jitter to avoid thundering herd
            # Narrate the backoff on the live activity line. `what` distinguishes a
            # rate-limited target call from a rate-limited judge call; the id comes
            # from the run-scoped ContextVar so this deep helper needs no new args.
            reason = "rate limit" if retry_after is not None else "transient error"
            set_activity(
                active_benchmark.get(), "ratelimit",
                f"{what} hit a {reason} — retry {attempt + 2}/{RETRY_MAX_ATTEMPTS} "
                f"in {delay:.0f}s",
            )
            await asyncio.sleep(delay)


async def send_prompt(target, prompt_text: str) -> tuple[str | None, str | None]:
    from pyrit.models import Message  # type: ignore

    # Rebuild the request each attempt — a Message can pick up conversation state
    # once sent, so a retry must start from a fresh one.
    def _mk():
        return target.send_prompt_async(
            message=Message.from_prompt(prompt=prompt_text, role="user")
        )

    try:
        responses = await retry_async(_mk, what="target send")
        text = responses[0].get_value() if responses else ""
        return text, None
    except Exception as e:  # noqa: BLE001
        # The `response` column holds what the TARGET said, and nothing else.
        # A vision-LLM diagnosis of why nothing arrived used to be written here
        # so the cell wasn't blank, which meant model prose shipped in the CSV
        # export under a column header promising target output. Callers read the
        # diagnosis from target.last_debug["vision_diagnosis"] into `note`.
        return None, f"{type(e).__name__}: {e}"
