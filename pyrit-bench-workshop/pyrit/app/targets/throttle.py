"""Per-target burst pause: after every N requests, rest X seconds.

Some hosted APIs cap requests-per-minute hard enough that a benchmark's steady
stream of prompts trips their 429s. Rather than fail those rows, a target can be
told to send a burst of N requests and then rest for X seconds before the next
burst. Both numbers are configured per-target on the Add-Target form
(``pause_after_n`` and ``pause_rest_s`` in the target's extra config) and
installed here by ``build_target``.

The pause is applied by wrapping the target's ``send_prompt_async`` — the single
method every attack kind funnels through. Static attacks reach it via
``retry.send_prompt``; PyRIT's multi-turn orchestrators (Crescendo, RedTeaming,
TAP, PAIR) call it directly, turn by turn. Wrapping it therefore throttles every
request the target ever makes, with one counter, regardless of attack.

The wrap is an instance-attribute override, not a subclass: the target keeps its
class, identity, ``configuration``, ``get_identifier()`` and every other thing
PyRIT introspects, so a multi-turn target is unaffected. (PyRIT's targets are
plain classes — they set instance attributes in ``__init__`` — so shadowing one
method on the instance is safe.)
"""
from __future__ import annotations

import asyncio
import time
from typing import Any

from ..activity import active_benchmark, set_activity


def throttle_slept_ms(target: Any) -> float:
    """Cumulative milliseconds this target has spent resting between bursts.

    Zero for a target with no throttle installed. A timed send site captures
    this before and after the send and subtracts the difference, so a rest is
    NEVER counted as target latency — a 60-second pause is not a 60-second
    response. See static_attack / dynamic_attack.
    """
    return getattr(target, "_throttle_slept_ms", 0.0)


def read_pause(extra: dict) -> tuple[int, float]:
    """Return ``(pause_after_n, pause_rest_s)`` from a target's extra config.

    Missing, blank or unparseable values disable the pause (returned as 0), so a
    malformed entry never crashes a run — it just means no throttling. Negative
    values are clamped to 0 for the same reason.
    """
    def _num(key: str, cast):
        try:
            return cast(str(extra.get(key, "")).strip())
        except (TypeError, ValueError):
            return cast(0)

    return max(0, _num("pause_after_n", int)), max(0.0, _num("pause_rest_s", float))


def install_throttle(target: Any, extra: dict) -> Any:
    """Wrap ``target.send_prompt_async`` to rest after every N requests.

    Returns the target unchanged unless BOTH ``pause_after_n`` >= 1 and
    ``pause_rest_s`` > 0 are set, so a target with no pause configured (every UI
    target, and any API target that left the fields blank) pays nothing.
    """
    every, rest = read_pause(extra)
    if every < 1 or rest <= 0:
        return target

    original = target.send_prompt_async
    state = {"sent": 0}

    async def _throttled(*args, **kwargs):
        # Rest once a full burst of N has gone out, BEFORE the next send starts.
        # Counting completed sends (rather than sleeping up front) means a run
        # that ends exactly on a burst boundary never sleeps on its way out.
        if state["sent"] and state["sent"] % every == 0:
            t0 = time.perf_counter()
            # Narrate the burst pause on the live activity line so a run that goes
            # quiet for `rest` seconds reads as "resting under the rate cap", not
            # as a hang. The benchmark id comes from the run-scoped ContextVar.
            set_activity(
                active_benchmark.get(), "throttle",
                f"sent {state['sent']} — resting {rest:.0f}s before the next burst",
            )
            await asyncio.sleep(rest)
            # Accumulate the real rest so the timed send sites can subtract it
            # from measured latency (see throttle_slept_ms). Measured, not
            # assumed, because the event loop may sleep slightly longer.
            target._throttle_slept_ms = throttle_slept_ms(target) + (
                time.perf_counter() - t0) * 1000
        state["sent"] += 1
        return await original(*args, **kwargs)

    target.send_prompt_async = _throttled
    return target
