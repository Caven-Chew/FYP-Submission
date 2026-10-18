"""Sending with retry/backoff, plus the consecutive-failure abort guard."""
from __future__ import annotations

import asyncio
import math
import time

from ..activity import active_benchmark, set_activity
from ..settings import RETRY_BASE_DELAY, RETRY_MAX_ATTEMPTS, RETRY_MAX_DELAY


class BenchmarkAborted(Exception):
    """Raised to stop a benchmark early once too many target calls fail in a row."""


class RequestAccountingError(RuntimeError):
    """Local persistence failed; never reinterpret it as a provider failure."""


def _observe(observer, method, *args, **kwargs):
    try:
        return getattr(observer, method)(*args, **kwargs)
    except Exception as exc:
        raise RequestAccountingError(
            "Local request history could not be saved. Execution stopped; inspect the request ledger before retrying."
        ) from exc


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


async def retry_async(make_coro, *, what: str = "LLM call", observer=None):
    """Bounded retry of local invocations; optional durable accounting.

    Persistence runs outside the provider exception handler. A ledger failure
    must never be interpreted as a reason to repeat an already sent request.
    """
    import random
    maximum = max(1, RETRY_MAX_ATTEMPTS)
    max_delay = RETRY_MAX_DELAY if math.isfinite(RETRY_MAX_DELAY) and RETRY_MAX_DELAY >= 0 else 60.0
    base_delay = RETRY_BASE_DELAY if math.isfinite(RETRY_BASE_DELAY) and RETRY_BASE_DELAY >= 0 else 1.0
    for attempt in range(maximum):
        row = _observe(observer, "begin", attempt + 1, maximum) if observer else None
        start = time.perf_counter()
        try:
            result = await make_coro()
        except Exception as exc:  # noqa: BLE001
            elapsed = max(0, int((time.perf_counter() - start) * 1000))
            retryable, retry_after = retryable_error(exc)
            again = retryable and attempt + 1 < maximum
            delay = None
            if again:
                if retry_after is None or not math.isfinite(retry_after) or retry_after < 0:
                    retry_after = min(max_delay, base_delay * (2 ** attempt))
                delay = max(0.0, min(max_delay, retry_after))
                delay = min(max_delay, delay + random.uniform(0, delay * 0.25))
            status_code = getattr(getattr(exc, "response", None), "status_code", None)
            if observer:
                _observe(observer, "finish", row, status="error", elapsed_ms=elapsed,
                                reason=type(exc).__name__,
                                http_status=status_code if isinstance(status_code, int) else None,
                                retryable=retryable,
                                delivery="response_received" if isinstance(status_code, int) else "unknown",
                                backoff_planned_ms=int(delay * 1000) if again else None)
            if not again:
                raise
            set_activity(active_benchmark.get(), "ratelimit",
                         f"{what} hit a transient error — retry {attempt + 2}/{maximum} in {delay:.0f}s")
            wait_start = time.perf_counter()
            try:
                await asyncio.sleep(delay)
            finally:
                if observer:
                    _observe(observer, "backoff", row, max(0, int((time.perf_counter() - wait_start) * 1000)))
        except BaseException:
            if observer:
                _observe(observer, "finish", row, status="interrupted", reason="call_cancelled",
                                delivery="unknown", elapsed_ms=max(0, int((time.perf_counter() - start) * 1000)))
            raise
        else:
            if observer:
                _observe(observer, "finish", row, status="returned", delivery="helper_returned",
                                elapsed_ms=max(0, int((time.perf_counter() - start) * 1000)))
            return result


async def send_prompt(target, prompt_text: str) -> tuple[str | None, str | None]:
    from pyrit.models import Message  # type: ignore

    # Rebuild the request each attempt — a Message can pick up conversation state
    # once sent, so a retry must start from a fresh one.
    def _mk():
        return target.send_prompt_async(
            message=Message.from_prompt(prompt=prompt_text, role="user")
        )

    try:
        from .request_history import observer_for
        responses = await retry_async(_mk, what="target send", observer=observer_for("target_send"))
        text = responses[0].get_value() if responses else ""
        return text, None
    except RequestAccountingError:
        raise
    except Exception as e:  # noqa: BLE001
        # The `response` column holds what the TARGET said, and nothing else.
        # A vision-LLM diagnosis of why nothing arrived used to be written here
        # so the cell wasn't blank, which meant model prose shipped in the CSV
        # export under a column header promising target output. Callers read the
        # diagnosis from target.last_debug["vision_diagnosis"] into `note`.
        return None, f"{type(e).__name__}: {e}"
