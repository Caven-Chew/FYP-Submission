"""One-shot connection test for a configured target."""
from __future__ import annotations

from ..targets import build_target

from .retry import send_prompt
from ..prompts import CONNECTION_TEST_EXPECT, CONNECTION_TEST_PROMPT


def test_target(t) -> dict:
    """Build a target and send one tiny prompt to verify it responds.

    Returns {ok, response, latency_ms, error}, plus `warning` (a recognisable
    capture fault) or `note` (a reply other than "pong"). Never raises — build/connect
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
    # fragment of the real reply, or on the user's own message echoed by a bad
    # selector. Two capture faults are recognisable and get a warning: the
    # prompt itself coming back (it contains "pong", so check it first) and a
    # strict prefix of "pong" (a streamed reply read too early). Any other
    # non-empty reply counts as connected with a neutral note: a scope-hardened
    # bot (the course's own mock included) rightly ignores "say pong"
    # (review finding #58). Warnings name form labels, not internal keys.
    if result["ok"]:
        got = " ".join((text or "").split()).lower()
        if " ".join(CONNECTION_TEST_PROMPT.split()).lower() in got:
            result["warning"] = (
                "The reply contains the test message itself, so the Response selector "
                "probably matched your own message, not your bot's reply. Run Capture "
                "from the page again and click the bot's reply."
            )
        elif CONNECTION_TEST_EXPECT in got:
            pass
        elif got and CONNECTION_TEST_EXPECT.startswith(got):
            result["warning"] = (
                f"The reply {text.strip()!r} is the beginning of {CONNECTION_TEST_EXPECT!r}, "
                "not the whole word: it was read while your bot was still typing. Replies "
                "in a run would be cut off the same way. Raise \"Wait for response — min "
                "(seconds)\" for this target."
            )
        else:
            shown = " ".join(text.split())
            shown = shown if len(shown) <= 200 else shown[:199] + "…"
            result["note"] = (f"Your bot answered: \"{shown}\" If this is your bot's real reply, "
                              "the target is set up.")
    if debug is not None:
        result["debug"] = debug
    return result
