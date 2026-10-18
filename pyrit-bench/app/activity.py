"""Live per-benchmark activity signal — "what is happening right now".

A benchmark runs in a daemon thread (see engine/runner.start_benchmark) while
the web thread serves the detail page and its ``status.json`` poll. The visible
``Benchmark.status`` ("queued"/"running"/"paused"/…) is coarse: it says a run is
*running* but not whether, this instant, it is generating an adversarial prompt,
waiting on the target, scoring a reply, or resting out a rate-limit. This module
carries that fine-grained phase between the two threads.

It mirrors engine/control.py exactly: a plain in-memory dict guarded by a lock,
shared because both threads live in one process, and deliberately NOT persisted —
a process restart kills the run thread, so there is nothing left to narrate. The
status endpoint reads the latest phase; the run loop writes it at each step.

Placed at the app root (next to settings.py) rather than under engine/ because
both ``engine`` and ``targets`` (the burst-pause throttle) write to it, and the
import DAG only allows both to depend *downward* on a shared root module — never
on each other.
"""
from __future__ import annotations

import threading
import time
from contextvars import ContextVar

# The benchmark whose run thread is currently executing, set once for the whole
# run in engine/runner. Lets deep helpers (retry backoff, the burst throttle)
# report activity without threading a benchmark id through every call — the same
# trick engine/llms.py uses for per-run LLM overrides. A ContextVar is inherited
# by tasks spawned on the run's event loop, so the throttle wrapper and the retry
# sleep both see it.
active_benchmark: ContextVar[int | None] = ContextVar("active_benchmark", default=None)

# Human-readable label for each phase key, shown verbatim in the UI. Keeping the
# label here (not at the call site) keeps the wording consistent across every
# place that sets a given phase.
PHASE_LABELS: dict[str, str] = {
    "queued": "Queued",
    "preparing": "Preparing run",
    "loading": "Loading prompts",
    "generating": "Generating prompt",
    "sending": "Sending prompt — waiting for reply",
    "judging": "Scoring reply",
    "attacking": "Running multi-turn attack",
    "ratelimit": "Rate-limited — backing off",
    "throttle": "Burst pause (per-target rate cap)",
    "waiting": "Pausing between requests",
    "paused": "Paused",
    "finishing": "Finishing up",
}

_lock = threading.Lock()
# benchmark_id -> {"phase", "detail", "since"}
_activity: dict[int, dict] = {}


def set_activity(benchmark_id: int | None, phase: str, detail: str | None = None) -> None:
    """Record what benchmark ``benchmark_id`` is doing right now.

    No-op when ``benchmark_id`` is None, so call sites that only have the
    ContextVar can pass ``active_benchmark.get()`` unconditionally. ``since`` is
    reset only when the phase actually changes, so the UI can show how long the
    current phase has lasted without it resetting on every repeated write.
    """
    if benchmark_id is None:
        return
    with _lock:
        prev = _activity.get(benchmark_id)
        changed = prev is None or prev.get("phase") != phase or prev.get("detail") != detail
        _activity[benchmark_id] = {
            "phase": phase,
            "detail": detail,
            "since": prev["since"] if (prev and not changed) else time.time(),
        }


def get_activity(benchmark_id: int) -> dict | None:
    """The current phase for one benchmark, or None if nothing is recorded.

    Returns a fresh dict with the label resolved and ``elapsed`` (seconds in the
    current phase) computed, so the caller can render it directly.
    """
    with _lock:
        a = _activity.get(benchmark_id)
        if not a:
            return None
        return {
            "phase": a["phase"],
            "label": PHASE_LABELS.get(a["phase"], a["phase"]),
            "detail": a.get("detail"),
            "elapsed": max(0.0, time.time() - a["since"]),
        }


def clear_activity(benchmark_id: int) -> None:
    """Forget a benchmark's activity (called when its run thread exits)."""
    with _lock:
        _activity.pop(benchmark_id, None)
