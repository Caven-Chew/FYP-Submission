"""Static attacks: dataset prompts, optionally converted, one turn each."""
from __future__ import annotations

import asyncio
import time
from ..activity import set_activity
from ..extensions import db
from ..models import Prompt, PromptStatus

from .control import wait_while_paused
from .pending_scores import PendingScores, close_after_reply, prepare_reply_for_close
from .converters import apply_converters, build_converters, is_content_policy_block
from .datasets import load_dataset_prompts
from .llms import build_converter_llm_from_spec
from .retry import BenchmarkAborted, send_prompt
from .scoring import (
    assessment_settings, detect_canary, build_response_decoder, looks_like_refusal, judge_response, severity_provenance
)
from ..settings import (
    ATTACKER_BACKUP_MODEL as ENV_ATTACKER_BACKUP_MODEL,
    CONVERTER_ON_BLOCK as ENV_CONVERTER_ON_BLOCK,
    MAX_CONVERTER_FAILURES, REQUEST_DELAY, attempts_for,
)
from ..targets.throttle import throttle_slept_ms


async def run_static_attack(bench, attack, target, judge, limit, guard, *, retry_prompt=None, resume=False, retry_failed=True, pending_scores=None) -> bool:
    """Run a static-dataset attack.

    Returns True when the attack reaches its end, False on an early stop. Totals are computed once, from the rows, by
    web.rollups.benchmark_totals — see the note in runner.py.
    Raises if dataset loading or converter setup fails — caller should isolate.
    `guard` is shared across the whole benchmark; if it trips this stops
    sending further prompts and returns whatever was already processed —
    never raises, so no partial progress is lost. The caller (`_run_attacks`)
    checks `guard.tripped` afterwards to decide whether to stop the whole
    benchmark.
    """
    if pending_scores is None:
        async with PendingScores() as pending:
            return await run_static_attack(
                bench, attack, target, judge, limit, guard,
                retry_prompt=retry_prompt, resume=resume, retry_failed=retry_failed,
                pending_scores=pending,
            )

    # An attack is one of:
    #   - single `converter` (one slug or chain), retried `attempts` times
    #     (best-of-N over stochastic rewrites); or
    #   - a `sweep`: a list of candidate converter chains. Each prompt is tried
    #     against each chain in turn and counts as broken if ANY lands. This
    #     turns "I picked a chain" into "the tool searched chains and found the
    #     one that works", and records which chain landed so the report can name
    #     it. attempts is ignored for a sweep (the chain list is the budget).
    sweep = attack.get("sweep")
    if sweep:
        plans = [(slug, (lambda sl=slug: build_converters(sl))) for slug in sweep]
    else:
        conv_slug = attack.get("converter")
        factory = (lambda: build_converters(conv_slug)) if conv_slug else (lambda: [])
        plans = [((conv_slug or ""), factory)] * attempts_for(attack)

    # Build every distinct chain once up front so an unknown slug or a missing
    # converter LLM raises here — the caller records one clear error row for the
    # attack instead of the same failure repeated for every prompt.
    for index, (_label, factory) in enumerate(plans):
        if retry_prompt is not None:
            if (index != retry_prompt.attempt or retry_prompt.status == PromptStatus.SEND_ERROR
                    or retry_prompt.can_rescore):
                continue
        factory()

    set_activity(bench.id, "loading", f"loading prompts for {attack.get('name', attack['slug'])}")
    if attack["dataset"] == "custom":
        # Students' own cases always run in full; the per-attack cap is for datasets.
        prompts = bench.custom_prompts
        if not prompts:
            raise ValueError("This custom prompt-list benchmark has no saved prompts.")
    else:
        prompts = await load_dataset_prompts(attack["dataset"], limit)
    tax = attack.get("taxonomy_category")
    name = attack.get("name", attack["slug"])
    # Which 1-5 scale this attack's rows are on. Default "harm"; extraction
    # attacks declare scoring="leak" because nothing harmful was requested and
    # the harm rubric scores a complete leak a 1. See scoring.RUBRICS.
    assessment = assessment_settings(bench, attack)
    rubric = "harm" if assessment.get("method") == "dual_v1_v2" else assessment["rubric"]
    n_seeds = len(prompts)
    # A converter that is broken is broken for every seed, so retrying it 30
    # times just fills the table with identical error rows — harmbench_math did
    # exactly that in benchmark 57. This counter abandons the ATTACK, not the
    # run: a converter fault says nothing about whether the target is up, so it
    # must never touch the shared target guard.
    consecutive_converter_failures = 0

    # Use the configured rewrite backup, preserving overrides on historical runs.
    cfg = bench.llm_config or {}
    backup_spec = (cfg.get("PROMPT_REWRITE_BACKUP_MODEL") or cfg.get("CONVERTER_BACKUP")
                   or ENV_ATTACKER_BACKUP_MODEL or None)
    on_block = (cfg.get("CONVERTER_ON_BLOCK") or ENV_CONVERTER_ON_BLOCK
                or "skip").strip().lower()

    if retry_prompt is not None:
        if retry_prompt.sequence >= len(prompts) or retry_prompt.attempt >= len(plans):
            raise ValueError("The original prompt or converter attempt is no longer available.")

    existing = {(p.sequence, p.attempt or 0): p for p in
                Prompt.query.filter_by(benchmark_id=bench.id, attack_slug=attack["slug"]).all()
                if p.prompt_text and p.status != PromptStatus.SETUP_ERROR} if resume else {}
    solved_seeds = {p.sequence for p in existing.values() if p.succeeded}
    selected_retry = retry_prompt

    def save_result(row):
        if retry_prompt is None:
            db.session.add(row)
        else:
            # Replace only this failed attempt, keeping its identity and group.
            for field in ("prompt_text", "converter_used", "response", "latency_ms",
                          "refused", "succeeded", "severity", "status", "verdict",
                          "severity_source", "latency_measured", "note", "error", "capture_ref",
                          "rubric", "canary_detected"):
                setattr(retry_prompt, field, getattr(row, field))
        # Commit each result as it is saved, as scripted conversations do, so
        # the run page shows progress and a crash or restart mid-attack keeps
        # every reply already collected for Resume to reuse (review finding #24).
        db.session.commit()
        return row if retry_prompt is None else retry_prompt

    for i, seed in enumerate(prompts):
        if selected_retry is not None and i != selected_retry.sequence:
            continue
        # Pause checkpoint: block here between seeds while the run is paused.
        await wait_while_paused(bench.id)
        group = f"{attack['slug']}#{i}"
        for attempt, (plan_label, factory) in enumerate(plans):
            if selected_retry is not None and attempt != selected_retry.attempt:
                continue
            if resume:
                retry_prompt = existing.get((i, attempt))
                if retry_prompt is not None and (not retry_failed or not retry_prompt.can_rerun):
                    continue
                # An earlier successful attempt means later missing slots were
                # deliberately skipped by best-of-N, not interrupted work.
                if retry_prompt is None and i in solved_seeds:
                    continue
            await wait_while_paused(bench.id)
            if retry_prompt is not None and retry_prompt.can_rescore:
                # The reply was captured; only the judge failed. Judge the
                # stored reply again and never contact the target (finding #3).
                set_activity(bench.id, "judging",
                             f"{name}: re-scoring the stored reply to seed {i + 1}/{n_seeds}")
                verdict, severity, succeeded = await judge_response(
                    judge, seed, retry_prompt.response,
                    decode=build_response_decoder(plan_label), **({"assessment_prompt": retry_prompt.prompt_text} if assessment.get("method") in {"dual_v1_v2", "checklist_v2"} else {}), **assessment)
                recovered = "recovered from the page by the vision model" in (retry_prompt.note or "")
                retry_prompt.verdict, retry_prompt.severity = verdict, severity
                retry_prompt.succeeded = succeeded
                retry_prompt.severity_source = severity_provenance(verdict, severity)
                retry_prompt.status = (PromptStatus.JUDGE_ERROR if severity is None and assessment.get("strict")
                                       else PromptStatus.RECOVERED if recovered else PromptStatus.OK)
                db.session.commit()
                if succeeded:
                    solved_seeds.add(i)
                continue
            label = (plan_label or "")[:60] or None
            # Rebuild the chain per attempt: the LLM-backed converters
            # (poetry / persuasion / tense) are stochastic, so each attempt
            # gets a fresh rewrite. For a sweep, each attempt is a different
            # chain entirely.
            text, conv_note, conv_declined, conv_error = seed, None, None, None
            conv_exc = None
            # A converter chain that rewrites the seed (poetry/persuasion/encode)
            # is real work — often an attacker-LLM call — so narrate it. A bare
            # prompt with no converter skips straight to "sending".
            if label:
                set_activity(bench.id, "generating",
                             f"{name}: rewriting seed {i + 1}/{n_seeds} via {label}")
            try:
                if retry_prompt is not None and retry_prompt.status == PromptStatus.SEND_ERROR:
                    text = retry_prompt.prompt_text
                else:
                    text, conv_note, conv_declined = await apply_converters(factory(), seed)
            except Exception as e:  # noqa: BLE001
                conv_error = f"converter-error: {type(e).__name__}: {e}"
                conv_exc = e

            # A content-policy block means the converter MODEL refused THIS seed
            # (e.g. OpenAI 400 "flagged for possible biological risk"), not a
            # structural fault. Retry the rewrite with the backup converter model;
            # if it rescues the seed the attack proceeds exactly as normal.
            content_block = bool(conv_error) and is_content_policy_block(conv_exc)
            if content_block and backup_spec:
                set_activity(bench.id, "generating",
                             f"{name}: primary converter content-blocked seed "
                             f"{i + 1}/{n_seeds} — retrying via backup {backup_spec}")
                backup = build_converter_llm_from_spec(backup_spec)
                if backup is not None:
                    try:
                        text, conv_note, conv_declined = await apply_converters(
                            build_converters(plan_label, converter_target=backup), seed)
                        conv_error, conv_exc, content_block = None, None, False
                    except Exception as e2:  # noqa: BLE001 — backup blocked too
                        conv_error = (f"converter-error: primary content-blocked and "
                                      f"backup {backup_spec} also failed: "
                                      f"{type(e2).__name__}: {e2}")
                        conv_exc = e2
                        content_block = True  # still per-prompt, not structural

            if conv_error or conv_declined:
                # A per-prompt block — the converter refused this seed, or a
                # content-policy block the backup could not rescue — is an absence
                # of data, NOT a structural converter fault. Recording the
                # converter's apology and scoring the target's "No problem!" reply
                # is exactly how benchmark 57 manufactured defensive wins, so
                # `prompt_text` keeps the seed we meant to attack with and the note
                # keeps why nothing was sent.
                per_prompt_block = bool(conv_declined) or content_block
                if content_block:
                    base_note = conv_declined or conv_note
                    detail = ("converter content-policy block on this seed"
                              + (f"; backup {backup_spec} also blocked" if backup_spec
                                 else "; no backup converter configured"))
                    note_txt = f"{base_note} | {detail}" if base_note else detail
                else:
                    note_txt = conv_declined or conv_note
                save_result(Prompt(
                    benchmark_id=bench.id, attack_slug=attack["slug"], sequence=i,
                    prompt_text=seed, converter_used=label, response="",
                    taxonomy_category=tax, attempt=attempt, attempt_group=group,
                    status=(PromptStatus.CONVERTER_REFUSED if per_prompt_block
                            else PromptStatus.CONVERTER_ERROR),
                    # Keep `error` empty for per-prompt blocks so failure filters
                    # skip them the way they skip a plain refusal; the detail lives
                    # in `note`. Only a structural error populates `error`.
                    error=(None if per_prompt_block else conv_error),
                    note=note_txt,
                    severity_source="none", latency_measured=False,
                ))
                # A per-prompt block never touches the abandonment guard. On a
                # content-policy block, honour the benchmark's on_block policy:
                # "stop" abandons the attack here, "skip" moves to the next seed.
                if per_prompt_block:
                    if content_block and on_block == "stop":
                        db.session.commit()
                        return False
                    continue
                # A structural converter error is likely broken for every seed, so
                # count consecutive failures and abandon the ATTACK (never the run)
                # once too many pile up.
                consecutive_converter_failures += 1
                if consecutive_converter_failures >= MAX_CONVERTER_FAILURES:
                    db.session.add(Prompt(
                        benchmark_id=bench.id, attack_slug=attack["slug"],
                        sequence=i, prompt_text="", response="",
                        taxonomy_category=tax,
                        status=PromptStatus.SETUP_ERROR,
                        severity_source="none", latency_measured=False,
                        error=(f"Attack abandoned after "
                               f"{consecutive_converter_failures} consecutive "
                               f"converter failures — the converter is broken for "
                               f"every prompt, not just this one. "
                               f"Last error: {conv_error}"),
                    ))
                    db.session.commit()
                    return False
                continue

            consecutive_converter_failures = 0
            # Exclude any burst-pause rest from the measured latency: if this
            # send crossed a throttle boundary, the target slept before it went
            # out, and that rest is not response time. Subtract only the rest
            # accrued across THIS send (before/after difference).
            slept0 = throttle_slept_ms(target)
            t0 = time.perf_counter()
            set_activity(bench.id, "sending",
                         f"{name}: seed {i + 1}/{n_seeds}"
                         + (f", attempt {attempt + 1}/{len(plans)}" if len(plans) > 1 else ""))
            prepare_reply_for_close(target)
            response, error = await send_prompt(target, text)
            latency = max(0, int((time.perf_counter() - t0) * 1000
                                 - (throttle_slept_ms(target) - slept0)))
            # A response the selector never matched, recovered from the page by
            # the vision model, is still a real response — but the row must say
            # so, or the results silently mix two different capture methods.
            dbg = getattr(target, "last_debug", None) or {}
            if response is not None and dbg.get("latency_ms") is not None:
                # A UI target times its own reply: send action to final reply text (4 Oct 2026).
                latency = dbg["latency_ms"]
            recovery_note = dbg.get("recovery_note")
            # Why nothing arrived, per the vision model. Commentary, not target
            # output — it belongs in `note`, never in `response`.
            diagnosis = dbg.get("vision_diagnosis")
            # Save capture before scoring, so the results page can display it
            # while the next independent prompt is sent.
            verdict, severity, succeeded = ("send-error" if error or not response else ""), None, False
            status = PromptStatus.SEND_ERROR if error or not response else PromptStatus.SCORING

            row = save_result(Prompt(
                benchmark_id=bench.id, attack_slug=attack["slug"], sequence=i,
                prompt_text=text, converter_used=label,
                response=response or "", latency_ms=latency,
                refused=looks_like_refusal(response), succeeded=succeeded,
                severity=severity, taxonomy_category=tax, rubric=rubric,
                canary_detected=detect_canary(bench, [response or ""]),
                attempt=attempt, attempt_group=group,
                status=status,
                # The judge's reasoning, kept so a severity can be shown and
                # defended rather than asserted. Previously computed, paid for
                # in judge tokens, and thrown away.
                verdict=verdict,
                severity_source=severity_provenance(verdict, severity),
                latency_measured=True,
                # Non-fatal commentary. Kept out of `error` so that filtering on
                # failures no longer discards a perfectly good scored row.
                note=recovery_note or conv_note or (
                    f"no response captured — {diagnosis}" if error and diagnosis else None),
                error=error,
                # Read immediately after the send, while last_debug still
                # describes THIS prompt — the target overwrites it on the next
                # one. None for non-browser targets and for sends that worked.
                capture_ref=(getattr(target, "last_debug", None) or {}).get("capture_ref"),
            ))

            score_task = None
            if response and not error:
                # Freeze this attempt's inputs: the loop and last_debug advance
                # while the judge is awaiting its provider.
                async def score_reply(row=row, seed=seed, text=text,
                                      response=response, plan_label=plan_label,
                                      recovery_note=recovery_note, seed_index=i):
                    try:
                        result = await judge_response(
                            judge, seed, response,
                            decode=build_response_decoder(plan_label),
                            **({"assessment_prompt": text} if assessment.get("method") in
                               {"dual_v1_v2", "checklist_v2"} else {}),
                            **assessment,
                        )
                        row.verdict, row.severity, row.succeeded = result
                        row.status = (PromptStatus.JUDGE_ERROR if row.severity is None and assessment.get("strict")
                                      else PromptStatus.RECOVERED if recovery_note else PromptStatus.OK)
                    except Exception as exc:
                        row.verdict = f"judge-error: {type(exc).__name__}: {exc}"
                        row.severity, row.succeeded = None, False
                        row.status = PromptStatus.JUDGE_ERROR
                    row.severity_source = severity_provenance(row.verdict, row.severity)
                    db.session.commit()
                    if row.succeeded:
                        solved_seeds.add(seed_index)
                    return row.succeeded

                closing = asyncio.create_task(close_after_reply(target))
                try:
                    score_task = await pending_scores.submit(score_reply)
                finally:
                    await closing
            else:
                await close_after_reply(target)

            try:
                guard.record(error)
            except BenchmarkAborted:
                db.session.commit()
                return False

            # A further best-of-N attempt depends on this verdict. Independent
            # seeds do not: let their captured replies be scored in the background.
            if score_task is not None and attempt + 1 < len(plans):
                succeeded = await score_task

            # Stop early once this seed has landed — best-of-N asks whether the
            # objective is reachable within the budget, not how many ways it is.
            if succeeded:
                solved_seeds.add(i)
                if not resume:
                    break
            if REQUEST_DELAY:
                set_activity(bench.id, "waiting", f"{name}: {REQUEST_DELAY:.0f}s between requests")
                await asyncio.sleep(REQUEST_DELAY)

    db.session.commit()
    return True
