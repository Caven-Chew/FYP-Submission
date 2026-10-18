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
from ..models import Benchmark, Prompt, PromptStatus, ConversationRetry
from ..targets import build_target
from ..targets.registry import build_conversation_target

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
from .pending_scores import PendingScores
from .scripted_attack import run_scripted_attack
from ..settings import (
    ATTACK_DELAY, MAX_CONSECUTIVE_TARGET_FAILURES,
    excluded_objective_categories, prompt_cap,
)

# Execution errors are unresolved prompt-level rows kept in this benchmark.
EXECUTION_ERRORS = (PromptStatus.SETUP_ERROR, PromptStatus.SEND_ERROR,
                    PromptStatus.CONVERTER_ERROR, PromptStatus.NO_CONVERSATION,
                    PromptStatus.INCOMPLETE, PromptStatus.JUDGE_ERROR)
# Target-side no-reply / send-no-evidence failures: enough to mark the run
# completed with target-side errors (rather than strictly failed).
TARGET_SIDE_EXECUTION_ERRORS = (PromptStatus.SEND_ERROR, PromptStatus.NO_CONVERSATION)


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
    was still alive after ``timeout``. The caller must retain the row on timeout.
    """
    control.request_cancel(benchmark_id)
    name = f"bench-{benchmark_id}"
    for t in threading.enumerate():
        if t.name == name and t.is_alive():
            t.join(timeout)
            return not t.is_alive()
    control.clear_cancel(benchmark_id)
    return True


def _execution_failure_summary(benchmark_id: int) -> tuple[str | None, bool]:
    """Return (status_detail, target_only_errors)."""
    errors = Prompt.query.filter(
        Prompt.benchmark_id == benchmark_id,
        Prompt.status.in_(EXECUTION_ERRORS),
    ).order_by(Prompt.id).all()
    if not errors:
        return None, False
    first = errors[0]
    reason = first.error or first.note or first.status
    detail = (
        f"Benchmark has {len(errors)} unresolved execution error(s). "
        f"Collected results are retained. First error ({first.attack_slug}): {reason}"
    )
    target_only = all(err.status in TARGET_SIDE_EXECUTION_ERRORS for err in errors)
    return detail, target_only


def finish_stopped(benchmark_id: int) -> None:
    """Keep completed work, update totals, and mark a manual stop as terminal."""
    db.session.commit()  # Static attacks may have finished rows pending a flush.
    b = db.session.get(Benchmark, benchmark_id)
    if b is None:
        return
    totals = b.compute_totals()
    b.total_prompts = totals["counted"]
    b.total_successes = totals["successes"]
    b.avg_latency_ms = totals["avg_latency_ms"]
    b.status = "stopped"
    b.status_detail = "Stopped manually. Collected results are retained; this is a partial run."
    b.finished_at = datetime.now(timezone.utc).replace(tzinfo=None)
    db.session.commit()


def _thread_main(app: Flask, benchmark_id: int, retry_id: int | None = None,
                 previous_status: str | None = None, recover: bool = False,
                 retry_failed: bool = True) -> None:
    with app.app_context():
        try:
            if recover:
                asyncio.run(_recover_async(benchmark_id, retry_failed=retry_failed))
            elif retry_id is None:
                asyncio.run(_run_async(benchmark_id))
            else:
                asyncio.run(_rerun_prompt(benchmark_id, retry_id, previous_status))
        except control.BenchmarkCancelled:
            finish_stopped(benchmark_id)
        except Exception as e:  # noqa: BLE001
            db.session.rollback()
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
    await control.wait_while_paused(benchmark_id)
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
    # at 3 prompts/attack without editing .env (and without the next research
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

    Failures are recorded per prompt/objective while remaining tests continue.
    The shared guard tracks consecutive errors without aborting the run; final
    status is failed if any execution errors remain. Manual Stop still cancels
    cooperatively at the next checkpoint.
    """
    guard = ConsecutiveFailureGuard(MAX_CONSECUTIVE_TARGET_FAILURES, stop_on_limit=False)
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
    scores = PendingScores()
    captured_attacks = []

    def mark_scored_attacks():
        for captured in captured_attacks:
            pending = Prompt.query.filter_by(benchmark_id=bench.id, attack_slug=captured,
                                             status=PromptStatus.SCORING).first()
            if pending is None and captured not in bench.completed_attacks:
                bench.completed_attacks_json = json.dumps(bench.completed_attacks + [captured])
        db.session.commit()

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

            if attack["kind"] != "static":
                await scores.drain()
                mark_scored_attacks()
            if attack["kind"] == "static":
                try:
                    await run_static_attack(bench, attack, target, judge, limit, guard, pending_scores=scores)
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
            elif attack["kind"] == "scripted":
                conv_target = None
                try:
                    conv_target = build_conversation_target(bench.target, require_history=any(len(turns) > 1 for turns in bench.custom_conversations))
                    _enable_captures(conv_target, bench.id)
                    await run_scripted_attack(bench, attack, conv_target, judge, limit, guard)
                except Exception as exc:
                    db.session.add(Prompt(benchmark_id=bench.id, attack_slug=slug,
                                          prompt_text="", status=PromptStatus.SETUP_ERROR,
                                          severity_source="none", error=f"{type(exc).__name__}: {exc}"))
                    db.session.commit()
                finally:
                    # This target belongs to the scripted attack alone; neither
                    # finally block below closes it, so a browser target's last
                    # conversation window would stay open (review finding #26).
                    aclose = getattr(conv_target, "aclose", None)
                    if aclose:
                        try:
                            await aclose()
                        except Exception:  # noqa: BLE001
                            pass  # Cleanup must not replace the original run outcome.
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
                    await control.wait_while_paused(bench.id)
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
                    await control.wait_while_paused(bench.id)
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

            await control.wait_while_paused(bench.id)
            # Attack runners stop early and report via guard.tripped rather than
            # raising, so results are already persisted before this check.
            if guard.tripped:
                raise BenchmarkAborted(guard.abort_message())

            # Capture finished cleanly. Mark complete only after its saved scores
            # settle, so a restart cannot skip a lost background judge task.
            # Committed immediately, so the marker is durable if the process is
            # killed at any later point in the run. Reached only when the guard
            # did NOT trip, so a target-unreachable abort leaves the attack
            # unmarked and it is redone on resume.
            captured_attacks.append(slug)
            mark_scored_attacks()

            # Pause between attacks for providers with strict per-minute caps.
            if ATTACK_DELAY:
                set_activity(bench.id, "waiting",
                             f"cooling down {ATTACK_DELAY:.0f}s before the next attack")
                await asyncio.sleep(ATTACK_DELAY)
    except BenchmarkAborted as exc:
        aborted_detail = str(exc)
    finally:
        await scores.drain()
        mark_scored_attacks()
        # This instance belongs to the attack loop; _run_async only closes the
        # original target. Release its last browser session on every exit,
        # including cooperative cancellation and unexpected attack failures.
        for mt in _mt_cache:
            aclose = getattr(mt, "aclose", None)
            if aclose:
                try:
                    await aclose()
                except Exception:  # noqa: BLE001
                    pass  # Cleanup must not replace the original run outcome.

    await control.wait_while_paused(bench.id)
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
        failure, target_only = _execution_failure_summary(bench.id)
        if failure:
            bench.status = "completed_with_target_errors" if target_only else "failed"
            bench.status_detail = failure
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


async def _rerun_prompt(benchmark_id: int, prompt_id: int, previous_status: str) -> None:
    bench = db.session.get(Benchmark, benchmark_id)
    prompt = db.session.get(Prompt, prompt_id)
    if not bench or not prompt or prompt.benchmark_id != benchmark_id or not prompt.can_rerun:
        raise ValueError("This prompt cannot be rerun.")
    active_benchmark.set(benchmark_id)
    llm_overrides.set(bench.llm_config or {})
    converter_llm_cache.set(None)
    target = None
    try:
        await control.wait_while_paused(benchmark_id)
        set_activity(benchmark_id, "preparing", "preparing failed prompt retry")
        attack = by_slug(prompt.attack_slug)
        dynamic = attack["kind"] == "dynamic"
        judge_only = (attack["kind"] == "scripted" and bench.llm_config.get("SCRIPTED_SCORE_UNIT") == "turn"
                      and prompt.can_rescore)
        if not judge_only:
            target = (build_conversation_target(bench.target, require_history=any(len(turns) > 1 for turns in bench.custom_conversations)) if attack["kind"] == "scripted"
                      else build_target(bench.target, multi_turn=True) if dynamic else build_target(bench.target))
            _enable_captures(target, benchmark_id)
        judge = build_judge_llm()
        if judge is None:
            raise ValueError("Configure a judge model before rerunning this prompt.")
        if attack["kind"] == "scripted":
            await run_scripted_attack(
                bench, attack, target, judge, max(prompt.sequence + 1, prompt_cap(bench, current_app.config["MAX_PROMPTS_PER_ATTACK"])),
                ConsecutiveFailureGuard(MAX_CONSECUTIVE_TARGET_FAILURES, stop_on_limit=False), retry_prompt=prompt)
        elif dynamic:
            await _rerun_conversation(bench, prompt, attack, target, judge)
        else:
            await run_static_attack(
                bench, attack, target, judge,
                max(prompt.sequence + 1, prompt_cap(bench, current_app.config["MAX_PROMPTS_PER_ATTACK"])),
                ConsecutiveFailureGuard(MAX_CONSECUTIVE_TARGET_FAILURES, stop_on_limit=False), retry_prompt=prompt,
            )
        totals = bench.compute_totals()
        bench.total_prompts = totals["counted"]
        bench.total_successes = totals["successes"]
        bench.avg_latency_ms = totals["avg_latency_ms"]
        # A retry that still has execution errors is failed, even if it began
        # from a manually stopped benchmark. A successful retry does not finish
        # attacks skipped by an earlier stop or abort.
        failure, target_only = _execution_failure_summary(bench.id)
        if failure:
            if previous_status == "completed_with_target_errors" and target_only:
                bench.status = "completed_with_target_errors"
            else:
                bench.status = "failed"
            bench.status_detail = failure
        else:
            # A successfully repaired, fully attempted run has no errors left.
            # Keep manually stopped/interrupted runs partial until recovery fills
            # their unattempted work.
            bench.status = ("completed" if previous_status in {"failed", "completed_with_target_errors"}
                            and set(bench.attack_slugs) <= set(bench.completed_attacks)
                            else previous_status)
            bench.status_detail = None
        db.session.commit()
    finally:
        if target is not None and getattr(target, "aclose", None):
            try:
                await target.aclose()
            except Exception:
                pass


def start_prompt_retry(app: Flask, benchmark_id: int, prompt_id: int,
                       previous_status: str) -> None:
    threading.Thread(
        target=_thread_main, args=(app, benchmark_id, prompt_id, previous_status),
        name=f"bench-{benchmark_id}", daemon=True,
    ).start()


async def _rerun_conversation(bench, prompt, attack, target, judge, *, guard=None):
    objectives = await resolve_objectives(
        attack, prompt_cap(bench, current_app.config["MAX_PROMPTS_PER_ATTACK"]),
        excluded_objective_categories(bench))
    selected = next(((i, row) for i, row in enumerate(objectives)
                     if (row.get("id") or f"obj-{i}") == prompt.objective_id), None)
    if selected is None:
        raise ValueError("The original objective is no longer available; existing results were kept.")
    objective_index, objective = selected
    objective = {**objective, "id": prompt.objective_id}
    roster = attacker_roster(bench)
    roster_index = prompt.attempt or 0
    if roster_index >= len(roster) or roster[roster_index]["label"] != prompt.converter_used:
        raise ValueError("The original attacker configuration is no longer available.")
    member = roster[roster_index]
    apply_attacker_override(member["config"])
    attacker = build_attacker_llm()
    if attacker is None:
        raise ValueError("Configure an attacker model before rerunning this conversation.")
    results = await run_dynamic_attack(
        attack=attack, target=target, attacker=attacker, judge=judge,
        objectives=[objective], benchmark_id=bench.id,
        guard=guard if guard is not None else ConsecutiveFailureGuard(MAX_CONSECUTIVE_TARGET_FAILURES, stop_on_limit=False))
    # Cancellation or setup failure must never erase the old conversation.
    await control.wait_while_paused(bench.id)
    if not results or any(r.get("objective_id") != prompt.objective_id for r in results):
        raise ValueError("Retry returned no valid conversation; existing results were kept.")
    originals = Prompt.query.filter_by(
        benchmark_id=bench.id, attack_slug=prompt.attack_slug,
        objective_id=prompt.objective_id, attempt=roster_index,
        converter_used=prompt.converter_used).all()
    snapshot = [{c.name: getattr(p, c.name) for c in Prompt.__table__.columns}
                for p in originals]
    if originals:
        db.session.add(ConversationRetry(
            benchmark_id=bench.id, attack_slug=prompt.attack_slug,
            objective_id=prompt.objective_id,
            original_rows_json=json.dumps(snapshot, default=str)))
    for r in results:
        # The isolated objective has local index zero; restore its position in
        # the full run, including the attack's internal best-of-N attempt.
        local_sequence = r.get("sequence", 0)
        offset = objective_index * 100 if r.get("status") == PromptStatus.OK else objective_index
        fields = {key: r.get(key) for key in (
            "prompt_text", "latency_ms", "succeeded", "severity", "taxonomy_category",
            "objective_id", "error", "verdict", "severity_source", "latency_measured",
            "note", "capture_ref")}
        fields["note"] = "Conversation rerun from the beginning." + (" " + fields["note"] if fields["note"] else "")
        db.session.add(Prompt(
            benchmark_id=bench.id, attack_slug=prompt.attack_slug,
            sequence=roster_index * 1_000_000 + local_sequence + offset,
            converter_used=member["label"], response=r.get("response") or "",
            refused=looks_like_refusal(r.get("response")), attempt=roster_index,
            attempt_group=r.get("attempt_group") or prompt.objective_id,
            status=r.get("status") or PromptStatus.OK, **fields))
    for old in originals:
        db.session.delete(old)
    db.session.flush()
    db.session.expire(bench, ["prompts", "retries"])
    # The caller commits the replacement, archive and refreshed totals together.


def start_benchmark_recovery(app: Flask, benchmark_id: int, *, retry_failed: bool = True) -> None:
    threading.Thread(target=_thread_main, args=(app, benchmark_id),
                     kwargs={"recover": True, "retry_failed": retry_failed}, name=f"bench-{benchmark_id}", daemon=True).start()


async def _recover_async(benchmark_id: int, *, retry_failed: bool = True) -> None:
    """Fill missing work, optionally retrying failures; preserve all other results."""
    bench = db.session.get(Benchmark, benchmark_id)
    active_benchmark.set(benchmark_id)
    llm_overrides.set(bench.llm_config or {})
    converter_llm_cache.set(None)
    bench.status_detail = None
    bench.finished_at = None
    if bench.started_at is None:
        bench.started_at = datetime.now(timezone.utc).replace(tzinfo=None)
    db.session.commit()
    targets = {}
    def get_target(dynamic=False):
        if dynamic not in targets:
            target = (build_conversation_target(bench.target, require_history=any(len(turns) > 1 for turns in bench.custom_conversations)) if dynamic == "scripted"
                      else build_target(bench.target, multi_turn=True) if dynamic else build_target(bench.target))
            _enable_captures(target, bench.id)
            targets[dynamic] = target
        return targets[dynamic]

    guard = ConsecutiveFailureGuard(MAX_CONSECUTIVE_TARGET_FAILURES, stop_on_limit=False)
    limit = prompt_cap(bench, current_app.config["MAX_PROMPTS_PER_ATTACK"])
    try:
        judge = build_judge_llm()
        if judge is None:
            raise ValueError("Configure a judge model before retrying this benchmark.")
        for slug in bench.attack_slugs:
            await control.wait_while_paused(bench.id)
            rows = Prompt.query.filter_by(benchmark_id=bench.id, attack_slug=slug).all()
            failures = [p for p in rows if p.status in EXECUTION_ERRORS or p.can_rerun]
            if slug in bench.completed_attacks and (not retry_failed or not failures):
                continue
            # A setup failure has no prompt boundary to resume from. Skipping
            # it means moving on to the next attack, retaining its error row.
            if not retry_failed and any(p.status == PromptStatus.SETUP_ERROR and not p.objective_id for p in rows):
                continue
            attack = by_slug(slug)
            if not attack:
                raise ValueError(f"Attack {slug} is no longer available; results were kept.")
            set_activity(bench.id, "preparing", f"{'retrying failed and unfinished tests' if retry_failed else 'continuing unattempted tests'}: {attack.get('name', slug)}")
            old_setup = [p for p in rows if p.status == PromptStatus.SETUP_ERROR and not p.objective_id]
            finished = True
            try:
                apply_attacker_override(None)
                if attack["kind"] == "static":
                    finished = await run_static_attack(
                        bench, attack, get_target(), judge, limit, guard, resume=True, retry_failed=retry_failed)
                elif attack["kind"] == "scripted":
                    finished = await run_scripted_attack(
                        bench, attack, get_target("scripted"), judge, limit, guard,
                        resume=True, retry_failed=retry_failed)
                elif attack["kind"] == "dynamic":
                    objectives = await resolve_objectives(attack, limit, excluded_objective_categories(bench))
                    if not objectives:
                        raise ValueError("No objectives available for this attack.")
                    for ridx, member in enumerate(attacker_roster(bench)):
                        for index, obj in enumerate(objectives):
                            await control.wait_while_paused(bench.id)
                            oid = obj.get("id") or f"obj-{index}"
                            group = [p for p in rows if p.objective_id == oid and (p.attempt or 0) == ridx
                                     and p.converter_used == member["label"]]
                            if group and (not retry_failed or not any(p.status in EXECUTION_ERRORS for p in group)):
                                continue
                            # A transient placeholder identifies missing work;
                            # it is never added to the database.
                            selected = group[0] if group else Prompt(
                                benchmark_id=bench.id, attack_slug=slug, objective_id=oid,
                                attempt=ridx, converter_used=member["label"], prompt_text=obj["objective"])
                            await _rerun_conversation(bench, selected, attack, get_target(True), judge, guard=guard)
                            db.session.commit()
                            if guard.tripped:
                                raise BenchmarkAborted(guard.abort_message())
                elif attack["kind"] == "azure_redteam":
                    if not retry_failed and rows:
                        # Azure cannot resume within an attempted scan.
                        continue
                    # Azure exposes a whole scan, not resumable conversations.
                    # Keep valid rows and fill only failed/missing sequence slots.
                    results = await run_azure_redteam_attack(attack=attack, target=get_target(), judge=judge, guard=guard)
                    if not results:
                        raise ValueError("Azure returned no results.")
                    kept = {p.sequence for p in rows if p.status not in EXECUTION_ERRORS}
                    for r in results:
                        if r["sequence"] in kept:
                            continue
                        fields = {k: v for k, v in r.items() if k in Prompt.__table__.columns.keys()
                                  and k not in {"id", "benchmark_id", "attack_slug"}}
                        db.session.add(Prompt(benchmark_id=bench.id, attack_slug=slug, **fields))
                    for p in rows:
                        if p.status in EXECUTION_ERRORS and any(r["sequence"] == p.sequence for r in results):
                            db.session.delete(p)
                    old_setup = []
                else:
                    raise ValueError(f"Unsupported attack kind: {attack['kind']}")
                for p in old_setup:
                    db.session.delete(p)
                db.session.commit()
            except BenchmarkAborted:
                raise
            except Exception as exc:
                db.session.rollback()
                # Keep prior results and one current setup error for this attack.
                setup = Prompt.query.filter_by(benchmark_id=bench.id, attack_slug=slug,
                                               status=PromptStatus.SETUP_ERROR, objective_id=None).first()
                if setup is None:
                    setup = Prompt(benchmark_id=bench.id, attack_slug=slug, prompt_text="",
                                   status=PromptStatus.SETUP_ERROR, severity_source="none", latency_measured=False)
                    db.session.add(setup)
                setup.error = f"{type(exc).__name__}: {exc}"
                db.session.commit()
                finished = False
            if guard.tripped:
                raise BenchmarkAborted(guard.abort_message())
            if finished is False and slug in bench.completed_attacks:
                bench.completed_attacks_json = json.dumps([s for s in bench.completed_attacks if s != slug])
                db.session.commit()
            elif finished is not False and slug not in bench.completed_attacks:
                bench.completed_attacks_json = json.dumps(bench.completed_attacks + [slug])
                db.session.commit()
            if ATTACK_DELAY:
                await asyncio.sleep(ATTACK_DELAY)
        await control.wait_while_paused(bench.id)
        failure, target_only = _execution_failure_summary(bench.id)
        unfinished = set(bench.attack_slugs) - set(bench.completed_attacks)
        if unfinished:
            bench.status = "failed"
            bench.status_detail = failure or "Some tests remain unfinished; collected results are retained."
        elif failure:
            bench.status = "completed_with_target_errors" if target_only else "failed"
            bench.status_detail = failure
        else:
            bench.status = "completed"
            bench.status_detail = None
    except BenchmarkAborted as exc:
        bench.status = "failed"
        bench.status_detail = f"Recovery ended because of execution errors: {exc}. Collected results are retained."
    finally:
        for target in targets.values():
            if getattr(target, "aclose", None):
                try:
                    await target.aclose()
                except Exception:
                    pass
    db.session.expire(bench, ["prompts"])
    totals = bench.compute_totals()
    bench.total_prompts = totals["counted"]
    bench.total_successes = totals["successes"]
    bench.avg_latency_ms = totals["avg_latency_ms"]
    bench.finished_at = datetime.now(timezone.utc).replace(tzinfo=None)
    db.session.commit()
