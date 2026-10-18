"""Cooperative pause / resume for a running benchmark.

A benchmark runs in a daemon thread (see runner.start_benchmark). Pausing can
never interrupt a target call that is already in flight, so it is cooperative:
the web thread flips a flag here, and the run loop checks that flag at safe
checkpoints — between prompts, between objectives, between attacks — and waits
there until it is resumed.

The flag is a plain in-memory set, shared by the web thread and the run thread
because they live in the same process. It is deliberately NOT persisted: a
process restart kills the run thread anyway, so there would be nothing left to
resume. The visible `Benchmark.status` ("paused"/"running") is what survives a
page reload; this set is only the live signal between the two threads.
"""
from __future__ import annotations

import asyncio
import threading

from ..activity import set_activity


class BenchmarkCancelled(BaseException):
    """Raised inside the run loop to stop a benchmark for good (not pause).

    Used when a *paused* benchmark is deleted: the run thread is still alive,
    parked at a checkpoint, so it must be told to unwind cleanly before the row
    is removed — otherwise it keeps committing Prompt rows against a benchmark_id
    that no longer exists (see benchmarks_delete).

    It derives from ``BaseException``, NOT ``Exception``, on purpose: the run
    loop wraps every attack in ``except Exception`` to isolate one bad attack
    from the rest, and a cancellation must sail straight through those handlers
    instead of being caught and logged as an attack error row. Only
    ``runner._thread_main`` catches it, at the very top of the thread.
    """


# benchmark ids that have been asked to pause. Guarded by _lock because the web
# thread mutates it while the run thread reads it.
_lock = threading.Lock()
_paused: set[int] = set()
# benchmark ids whose run must stop entirely (paused-then-deleted). Separate
# from _paused: a cancel must survive a resume and can never be "un-cancelled".
_cancelled: set[int] = set()

#: How often a paused run wakes to re-check whether it has been resumed.
_POLL_SECONDS = 1.0


def request_pause(benchmark_id: int) -> None:
    """Ask the run loop to pause at its next checkpoint. Idempotent."""
    with _lock:
        _paused.add(benchmark_id)


def request_resume(benchmark_id: int) -> None:
    """Let a paused run continue (and clear the flag for a finished run)."""
    with _lock:
        _paused.discard(benchmark_id)


def is_paused(benchmark_id: int) -> bool:
    with _lock:
        return benchmark_id in _paused


def request_cancel(benchmark_id: int) -> None:
    """Ask the run loop to stop this benchmark for good at its next checkpoint.

    Clears any pause too, so a paused loop wakes immediately, leaves its wait,
    and hits the cancel check rather than sleeping on. Idempotent.
    """
    with _lock:
        _cancelled.add(benchmark_id)
        _paused.discard(benchmark_id)


def is_cancelled(benchmark_id: int) -> bool:
    with _lock:
        return benchmark_id in _cancelled


def clear_cancel(benchmark_id: int) -> None:
    """Drop a benchmark's cancel flag (called when its run thread exits)."""
    with _lock:
        _cancelled.discard(benchmark_id)


async def wait_while_paused(benchmark_id: int) -> None:
    """Block here for as long as this benchmark is paused.

    Called at checkpoints in the run loop. Returns immediately when the
    benchmark is not paused, so it is cheap to call once per prompt. While
    paused it yields control back to the event loop every ``_POLL_SECONDS`` so
    the thread stays responsive to a resume.

    Raises ``BenchmarkCancelled`` if the benchmark has been cancelled (its row is
    being deleted), stopping the run right here at the checkpoint — before the
    next prompt is sent or written, so no orphan row is ever committed.
    """
    while is_paused(benchmark_id):
        if is_cancelled(benchmark_id):
            raise BenchmarkCancelled(benchmark_id)
        # Narrate the pause so the live activity line agrees with the status pill
        # (it will re-report itself each poll; set_activity only resets the
        # "since" clock on a real phase change, so the elapsed timer keeps ticking).
        set_activity(benchmark_id, "paused")
        await asyncio.sleep(_POLL_SECONDS)
    # Also catch a cancel arriving while NOT paused (e.g. a running benchmark),
    # so this one checkpoint call serves both cases.
    if is_cancelled(benchmark_id):
        raise BenchmarkCancelled(benchmark_id)
