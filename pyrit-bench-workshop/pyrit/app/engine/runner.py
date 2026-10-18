"""Benchmark orchestration: the background thread that runs a benchmark end to end."""
from __future__ import annotations

import asyncio
import json
import threading
import traceback
from datetime import datetime, timezone
from typing import Any
from flask import Flask, current_app
from ..activity import active_benchmark, clear_activity, set_activity
from ..artifacts import benchmark_dir, prune
from ..attacks import by_slug
from ..extensions import db
from ..models import Benchmark, Prompt, PromptStatus
from ..targets import build_target

from . import control
from .azure_attack import run_azure_redteam_attack
from .datasets import resolve_objectives
from .dynamic_attack import run_dynamic_attack
from .llms import (
    apply_attacker_override, attacker_roster, converter_llm_cache, llm_overrides,
    build_attacker_llm, build_judge_llm
)
from .retry import BenchmarkAborted, ConsecutiveFailureGuard
from .scoring import looks_like_refusal
from .static_attack import run_static_attack
from ..settings import (
    ATTACK_DELAY, MAX_CONSECUTIVE_TARGET_FAILURES,
    excluded_objective_categories, prompt_cap,
)


# ---------- main entry points --------------------------------------------

def _strip_startup_note(detail: str | None) -> str | None:
    """Remove the transient "[startup] … interrupted …" note from a status_detail.

    _mark_interrupted_benchmarks (app/__init__) appends a single "[startup] …"
    line to status_detail when a restart orphans a run. That note is only true
    until the run is resumed; once it relaunches it is stale, so this drops any
    line beginning with "[startup]" and returns None if nothing else remains.
    """
    if not detail:
        return detail
    kept = [ln for ln in detail.splitlines() if not ln.lstrip().startswith("[startup]")]
    return "\n".join(kept).strip() or None


def start_benchmark(app: Flask, benchmark_id: int) -> None:
    threading.Thread(
        target=_thread_main, args=(app, benchmark_id),
        name=f"bench-{benchmark_id}", daemon=True,
    ).start()


def cancel_and_join(benchmark_id: int, timeout: float = 10.0) -> bool:
    """Stop a benchmark's run thread and wait for it to unwind.

    Used before deleting a paused benchmark: signal the cancel, wake the parked
    loop (request_cancel clears the pause), then join the ``bench-<id>`` thread
    so it has stopped committing Prompt rows before the row is removed. The run
    loop only checks the cancel at checkpoints (between prompts/objectives), so a
    call already in flight finishes first — the same cooperative model as pause.

    Returns True if no live thread remains (safe to delete), False if the thread
    was still alive after ``timeout`` (a target call is taking longer than that;
    the caller can still delete, and the cancel flag keeps the thread from
    writing anything for the next prompt).
    """
    control.request_cancel(benchmark_id)
    name = f"bench-{benchmark_id}"
    for t in threading.enumerate():
        if t.name == name and t.is_alive():
            t.join(timeout)
            return not t.is_alive()
    return True


def _thread_main(app: Flask, benchmark_id: int) -> None:
    with app.app_context():
        try:
            asyncio.run(_run_async(benchmark_id))
        except control.BenchmarkCancelled:
            # The benchmark was cancelled (deleted while paused). Its row is
            # being removed by the web thread, so write nothing — just unwind.
            db.session.rollback()
        except Exception as e:  # noqa: BLE001
            b = db.session.get(Benchmark, benchmark_id)
            if b:
                b.status = "failed"
                b.status_detail = f"{e}\n{traceback.format_exc()}"
                b.finished_at = datetime.now(timezone.utc).replace(tzinfo=None)
                db.session.commit()
        finally:
            # Drop any lingering pause / cancel flag once the thread exits, so the
            # id is never left flagged with no thread behind it (and a reused id
            # can't inherit a stale pause or cancel).
            control.request_resume(benchmark_id)
            control.clear_cancel(benchmark_id)
            # Forget the live activity too — the run is over, so there is nothing
            # left to narrate and a reused id must not inherit a stale phase.
            clear_activity(benchmark_id)


async def _run_async(benchmark_id: int) -> None:
    bench = db.session.get(Benchmark, benchmark_id)
    if not bench:
        return
    # Bind this run to the activity ContextVar so deep helpers (retry backoff,
    # the burst throttle) can report their phase without being handed the id.
    active_benchmark.set(benchmark_id)
    set_activity(benchmark_id, "preparing", "building target and judge")
    # Apply this benchmark's per-run LLM overrides (attacker/judge provider,
    # model, key) for the whole run. Set on the ContextVar so build_judge_llm /
    # build_attacker_llm / the converters all see it without threading it
    # through every call. Reset in the finally below.
    llm_overrides.set(bench.llm_config or {})
    converter_llm_cache.set(None)
    bench.status = "running"
    # This run supersedes the transient "[startup] … interrupted … press Resume"
    # note that _mark_interrupted_benchmarks left on the row after a restart.
    # Strip it now the run is relaunching: otherwise a run that goes on to
    # complete cleanly still advertises that it was interrupted (which is exactly
    # why benchmark 71 read as unfinished despite finishing), and a run that
    # fails would show the stale restart message instead of the real failure.
    bench.status_detail = _strip_startup_note(bench.status_detail)
    # Preserve the original start time across a resume, so a resumed run's
    # duration covers the whole run rather than just the resumed tail. On a fresh
    # run started_at is None and gets set here as before.
    if bench.started_at is None:
        bench.started_at = datetime.now(timezone.utc).replace(tzinfo=None)
    db.session.commit()

    # Retention is enforced at the start of a run, not the end: pruning here
    # means the files from the run you are about to replace are still on disk
    # while you set it up, and the sweep costs nothing when nothing is stale.
    prune()

    target = build_target(bench.target)
    _enable_captures(target, bench.id)
    judge = build_judge_llm()
    if judge is None:
        bench.status = "failed"
        bench.status_detail = (
            "JUDGE_PROVIDER is not configured. Set JUDGE_PROVIDER, JUDGE_MODEL and "
            "JUDGE_API_KEY_ENV in .env (the judge LLM scores each target response)."
        )
        bench.finished_at = datetime.now(timezone.utc).replace(tzinfo=None)
        db.session.commit()
        return

    # Per-benchmark cap wins over the .env default, so a workshop preset can run
    # at 2 prompts/attack without editing .env (and without the next research
    # run silently inheriting 3).
    limit = prompt_cap(bench, current_app.config["MAX_PROMPTS_PER_ATTACK"])
    try:
        await _run_attacks(bench, target, judge, limit)
    finally:
        # A UI browser target may still have a tab open for the last
        # conversation's final turn (BrowserUITarget now keeps the browser
        # session alive across turns of one conversation — see providers.py).
        # Close it here so it doesn't stay running for the rest of the process.
        aclose = getattr(target, "aclose", None)
        if aclose:
            try:
                await aclose()
            except Exception:  # noqa: BLE001
                pass


def _enable_captures(target: Any, benchmark_id: int) -> None:
    """Point a target's failure captures at this benchmark's folder.

    Only BrowserUITarget defines `capture_dir`; for API and HTTP targets there
    is no page to photograph, so this is a no-op by design rather than an error.
    """
    if hasattr(target, "capture_dir"):
        target.capture_dir = str(benchmark_dir(benchmark_id))


async def _run_attacks(bench: Benchmark, target: Any, judge: Any, limit: int) -> None:
    """Run every configured attack for a benchmark, updating rollups as it goes.

    A `ConsecutiveFailureGuard` is shared across every attack so that once the
    target looks unreachable (see its docstring) the whole benchmark stops
    instead of burning through every remaining prompt/objective/attack against
    a dead target. Whatever was already recorded stays recorded — only the
    attacks that hadn't started yet are skipped.
    """
    guard = ConsecutiveFailureGuard(MAX_CONSECUTIVE_TARGET_FAILURES)
    aborted_detail: str | None = None

    # Multi-turn strategies (Crescendo/RedTeaming/TAP/PAIR) introspect the
    # target for `configuration`, `get_identifier()` and the
    # `supports_editable_history` capability, so they need a real PyRIT
    # PromptChatTarget rather than the generic HTTP shim `target` may be.
    # Built lazily and cached: a benchmark with only static attacks never pays
    # for it, and never fails on a provider that has no SDK target class.
    _mt_cache: list[Any] = []

    def multi_turn_target() -> Any:
        if not _mt_cache:
            mt = build_target(bench.target, multi_turn=True)
            # Multi-turn attacks get their own target instance, so it needs the
            # capture folder set too — otherwise every Crescendo/TAP failure
            # would silently skip capture while static attacks captured fine.
            _enable_captures(mt, bench.id)
            _mt_cache.append(mt)
        return _mt_cache[0]

    # objective_id -> reason, for objectives where the attacker model looked
    # like it self-censored instead of attacking (see attacker_looks_weak).
    weak_attacker: dict[str, str] = {}

    try:
        for slug in bench.attack_slugs:
            # Resume support (per-attack granularity). An attack recorded as
            # completed on an earlier, interrupted run is skipped whole. For an
            # attack that did NOT finish, any partial rows it left behind are
            # cleared first, so redoing it from the start can't double-count.
            # On a fresh run completed_attacks is empty and no rows exist yet, so
            # both are no-ops — the same path serves fresh runs and resumes.
            if slug in bench.completed_attacks:
                continue
            Prompt.query.filter_by(benchmark_id=bench.id, attack_slug=slug).delete()
            db.session.commit()

            # Pause checkpoint between attacks. A finer one sits inside each
            # attack (see static_attack / dynamic_attack), so a pause takes
            # effect within a prompt or two rather than only at attack
            # boundaries; this one covers the gap between attacks and the
            # Azure path, which runs as one opaque call.
            await control.wait_while_paused(bench.id)

            attack = by_slug(slug)
            if not attack:
                continue
            set_activity(bench.id, "preparing", f"starting {attack.get('name', slug)}")

            if attack["kind"] == "static":
                try:
                    await run_static_attack(bench, attack, target, judge, limit, guard)
                except Exception as exc:  # noqa: BLE001
                    # Isolate failures so one bad attack doesn't kill the whole benchmark.
                    db.session.add(Prompt(
                        benchmark_id=bench.id, attack_slug=slug, sequence=0,
                        prompt_text="", response="",
                        status=PromptStatus.SETUP_ERROR,
                        severity_source="none", latency_measured=False,
                        error=f"{type(exc).__name__}: {exc}",
                    ))
                    db.session.commit()
            elif attack["kind"] == "dynamic":
                objectives = await resolve_objectives(
                    attack, limit, excluded_objective_categories(bench))
                if not objectives:
                    db.session.add(Prompt(
                        benchmark_id=bench.id, attack_slug=slug, sequence=0,
                        prompt_text="", response="",
                        status=PromptStatus.SETUP_ERROR,
                        severity_source="none", latency_measured=False,
                        taxonomy_category=attack.get("taxonomy_category"),
                        error="No objectives available (app/data/objectives.json missing and dataset fallback empty)",
                    ))
                    db.session.commit()
                    continue

                # Attacker roster: run this attack once per attacker model and
                # union the outcomes per objective (an objective counts as broken
                # if ANY attacker breaks it). Attacker capability is the dominant
                # driver of success, so trying several is a cheap multiplier on
                # weaknesses found — and `converter_used` records which attacker
                # actually landed each one. Empty roster => single pass on the
                # benchmark's configured / .env attacker (unchanged behaviour).
                for ridx, ratk in enumerate(attacker_roster(bench)):
                    label = ratk["label"]
                    apply_attacker_override(ratk["config"])
                    attacker = build_attacker_llm()
                    if attacker is None:
                        db.session.add(Prompt(
                            benchmark_id=bench.id, attack_slug=slug, sequence=0,
                            prompt_text="", response="",
                            status=PromptStatus.SETUP_ERROR,
                            severity_source="none", latency_measured=False,
                            taxonomy_category=attack.get("taxonomy_category"),
                            converter_used=label,
                            error="No attacker LLM configured — set an attacker model "
                                  "(benchmark form or ATTACKER_PROVIDER in .env)",
                        ))
                        db.session.commit()
                        continue
                    try:
                        results = await run_dynamic_attack(
                            attack=attack, target=multi_turn_target(),
                            attacker=attacker, judge=judge,
                            objectives=objectives, guard=guard,
                            benchmark_id=bench.id,
                        )
                    except Exception as exc:  # noqa: BLE001
                        db.session.add(Prompt(
                            benchmark_id=bench.id, attack_slug=slug, sequence=0,
                            prompt_text="", response="",
                            status=PromptStatus.SETUP_ERROR,
                            severity_source="none", latency_measured=False,
                            taxonomy_category=attack.get("taxonomy_category"),
                            converter_used=label,
                            error=f"{type(exc).__name__}: {exc}",
                        ))
                        db.session.commit()
                        continue
                    for r in results:
                        db.session.add(Prompt(
                            benchmark_id=bench.id, attack_slug=slug,
                            sequence=ridx * 1_000_000 + r["sequence"],
                            prompt_text=r["prompt_text"], converter_used=label,
                            response=r["response"] or "", latency_ms=r.get("latency_ms"),
                            refused=looks_like_refusal(r.get("response")),
                            succeeded=r["succeeded"], severity=r.get("severity"),
                            taxonomy_category=r.get("taxonomy_category"),
                            objective_id=r.get("objective_id"),
                            # attempt = roster index so best-of-N unions across
                            # attackers and the reach curve reads as "objectives
                            # broken vs number of attacker models tried".
                            attempt=ridx,
                            attempt_group=r.get("attempt_group") or r.get("objective_id"),
                            error=r.get("error"),
                            # Provenance carried straight through from the attack runner,
                            # the only place that knows whether a severity was judged or
                            # invented, and whether a latency was timed or divided out of
                            # one objective's total.
                            status=r.get("status") or PromptStatus.OK,
                            verdict=r.get("verdict"),
                            severity_source=r.get("severity_source"),
                            latency_measured=r.get("latency_measured"),
                            note=r.get("note"),
                            capture_ref=r.get("capture_ref"),
                        ))
                        if r.get("attacker_weak"):
                            weak_attacker[str(r.get("objective_id"))] = r["attacker_weak"]
                    db.session.commit()
                    if guard.tripped:
                        break
                # Restore the benchmark-level attacker override for later attacks.
                apply_attacker_override(None)

            elif attack["kind"] == "azure_redteam":
                try:
                    results = await run_azure_redteam_attack(
                        attack=attack, target=target, judge=judge, guard=guard,
                    )
                except Exception as exc:  # noqa: BLE001
                    db.session.add(Prompt(
                        benchmark_id=bench.id, attack_slug=slug, sequence=0,
                        prompt_text="", response="",
                        status=PromptStatus.SETUP_ERROR,
                        severity_source="none", latency_measured=False,
                        taxonomy_category=attack.get("taxonomy_category"),
                        error=f"{type(exc).__name__}: {exc}",
                    ))
                    db.session.commit()
                    continue
                for r in results:
                    db.session.add(Prompt(
                        benchmark_id=bench.id, attack_slug=slug,
                        sequence=r["sequence"],
                        prompt_text=r["prompt_text"], converter_used=None,
                        response=r["response"] or "", latency_ms=r.get("latency_ms"),
                        refused=looks_like_refusal(r.get("response")),
                        succeeded=r["succeeded"], severity=r.get("severity"),
                        taxonomy_category=r.get("taxonomy_category"),
                        objective_id=r.get("objective_id"),
                        attempt=r.get("attempt") or 0,
                        attempt_group=r.get("attempt_group") or r.get("objective_id"),
                        error=r.get("error"),
                        # Provenance carried straight through from the attack runner,
                        # the only place that knows whether a severity was judged or
                        # invented, and whether a latency was timed or divided out of
                        # one objective's total.
                        status=r.get("status") or PromptStatus.OK,
                        verdict=r.get("verdict"),
                        severity_source=r.get("severity_source"),
                        latency_measured=r.get("latency_measured"),
                        note=r.get("note"),
                        capture_ref=r.get("capture_ref"),
                    ))
                db.session.commit()

            # Attack runners stop early and report via guard.tripped rather than
            # raising, so results are already persisted before this check.
            if guard.tripped:
                raise BenchmarkAborted(guard.abort_message())

            # This attack finished cleanly (it did not abort the run). Record it
            # so a Resume after a restart skips it instead of redoing its work.
            # Committed immediately, so the marker is durable if the process is
            # killed at any later point in the run. Reached only when the guard
            # did NOT trip, so a target-unreachable abort leaves the attack
            # unmarked and it is redone on resume.
            bench.completed_attacks_json = json.dumps(bench.completed_attacks + [slug])
            db.session.commit()

            # Pause between attacks for providers with strict per-minute caps.
            if ATTACK_DELAY:
                set_activity(bench.id, "waiting",
                             f"cooling down {ATTACK_DELAY:.0f}s before the next attack")
                await asyncio.sleep(ATTACK_DELAY)
    except BenchmarkAborted as exc:
        aborted_detail = str(exc)

    # Counted ONCE, from the rows, by the same code the pages use. There used
    # to be a second set of counters accumulated here as the run progressed,
    # under subtly different rules — benchmark 57 stored total_prompts=240
    # against 244 actual rows, and neither number excluded the 37 rows where
    # nothing ever reached the target.
    set_activity(bench.id, "finishing", "computing totals")
    totals = bench.compute_totals()
    bench.total_prompts = totals["counted"]
    bench.total_successes = totals["successes"]
    bench.avg_latency_ms = totals["avg_latency_ms"]
    if aborted_detail:
        bench.status = "failed"
        # Point at whatever actually broke. Sending someone to check their
        # endpoint and credentials when a file went missing on their own disk
        # costs an afternoon — that is exactly how benchmark 57 read.
        if guard.last_error_is_local:
            fix = ("Results already collected before the stop are kept below. This is a "
                   "problem on this machine, not with the target — check the Python "
                   "environment (a missing package, browser binary, or file) and re-run.")
        else:
            fix = ("Results already collected before the stop are kept below. Fix the "
                   "target (endpoint, credentials, UI selectors) and re-run.")
        bench.status_detail = (
            f"Benchmark stopped early to avoid wasting time and quota: "
            f"{aborted_detail}\n\n{fix}"
        )
    else:
        bench.status = "completed"

    if weak_attacker:
        sample = next(iter(weak_attacker.values()))
        note = (
            f"[attacker warning] On {len(weak_attacker)} objective(s) the adversarial "
            f"model appears to have self-censored rather than attacking "
            f"(e.g. {sample}). A low success rate from those objectives is NOT "
            f"evidence that the target is robust — the attack was never really "
            f"mounted. Consider a less safety-trained ATTACKER_MODEL, and read the "
            f"adversarial turns before drawing conclusions."
        )
        bench.status_detail = f"{bench.status_detail}\n\n{note}" if bench.status_detail else note

    bench.finished_at = datetime.now(timezone.utc).replace(tzinfo=None)
    db.session.commit()
