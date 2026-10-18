"""One-shot connection test for a configured target."""
from __future__ import annotations

from ..targets import build_target

from .retry import send_prompt
from ..prompts import CONNECTION_TEST_EXPECT, CONNECTION_TEST_PROMPT


def test_target(t) -> dict:
    """Build a target and send one tiny prompt to verify it responds.

    Returns {ok, response, latency_ms, error}. Never raises — build/connect
    failures are captured as the `error` string so the caller can render a
    friendly hint. Works for any provider, including the Playwright UI target
    (which will briefly open a browser).
    """
    import asyncio as _asyncio
    import time as _time

    async def _run():
        target = build_target(t)            # may raise (unknown provider, bad config)
        # A manual connection test of a browser/UI target should be watchable:
        # force the window visible and linger a few seconds so the user can see
        # the page being driven (and inspect it on a selector failure). Benchmark
        # runs are unaffected — they call send_prompt directly, never test_target.
        linger = 0
        if hasattr(target, "headless"):     # BrowserUITarget
            target.headless = False
            target.linger_ms = max(getattr(target, "linger_ms", 0), 4000)
            linger = target.linger_ms
        t0 = _time.perf_counter()
        try:
            text, error = await send_prompt(target, CONNECTION_TEST_PROMPT)
        finally:
            # BrowserUITarget now keeps its browser session open across turns
            # of the same conversation (for multi-turn attacks) instead of
            # closing after every send — a one-shot test is its own whole
            # conversation, so close it explicitly here (this is what applies
            # the `linger` pause above before the window disappears).
            aclose = getattr(target, "aclose", None)
            if aclose:
                await aclose()
        # Don't count the inspection pause in the reported latency.
        elapsed = int((_time.perf_counter() - t0) * 1000) - linger
        # BrowserUITarget only — what response_sel actually collected, so the
        # connection test can show it back for debugging a bad capture.
        debug = getattr(target, "last_debug", None)
        return text, error, max(elapsed, 0), debug

    try:
        text, error, latency, debug = _asyncio.run(_run())
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "response": "", "latency_ms": None,
                "error": f"{type(e).__name__}: {e}"}

    result = {
        "ok": bool(text) and not error,
        "response": text or "",
        "latency_ms": latency,
        "error": error,
    }
    # "We got a non-empty string back" is a weak assertion: it passes on a
    # fragment of the real reply, or on a stray element the selector happened
    # to match. The test prompt asks for a specific word precisely so the
    # answer can be checked. A mismatch is not a hard failure — a small or
    # poorly-instructed model may legitimately ignore the instruction — but it
    # is always worth surfacing, because the far more common cause is a
    # capture problem that would silently corrupt every row of a benchmark.
    if result["ok"]:
        got = (text or "").strip().lower()
        if CONNECTION_TEST_EXPECT not in got:
            if got and CONNECTION_TEST_EXPECT.startswith(got):
                result["warning"] = (
                    f"The reply {text.strip()!r} is the beginning of {CONNECTION_TEST_EXPECT!r}, "
                    "not the whole word — it was captured while the target was still "
                    "streaming. Benchmark responses from this target would be truncated "
                    "the same way. Raise ui_stream_settle_ms (or ui_wait_ms) for it."
                )
            else:
                result["warning"] = (
                    f"The reply does not contain {CONNECTION_TEST_EXPECT!r}, which the test "
                    "prompt asked for. Either the response selector matched the wrong "
                    "element, or this target ignores simple instructions. Open the debug "
                    "block below and check what the selector actually collected."
                )
    if debug is not None:
        result["debug"] = debug
    return result
