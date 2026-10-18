"""Static judge-only recovery on a persisted request; never calls a target."""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import threading
import time

from ..attacks import by_slug
from ..activity import active_benchmark, clear_activity, set_activity
from ..extensions import db
from ..models import Benchmark, Prompt, ScoringEvaluation, _utcnow
from ..settings import SUCCESS_SEVERITY
from .llms import build_judge_llm, llm_overrides, resolve_role
from .request_history import request_history
from .retry import RequestAccountingError
from .scoring import (JUDGE_PARSER_VERSION, build_response_decoder, judge_request,
                      render_judge_request, severity_provenance)

class RecoveryUnavailable(ValueError):
    """A fixed, credential-free explanation of an unavailable saved contract."""


TERMINAL = ("completed", "failed", "completed_with_target_errors", "stopped", "interrupted")


def text_hash(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def endpoint_identity(judge):
    # Preserve equality of the configurable endpoint without storing its URL.
    return (text_hash(os.getenv("LLM_ENDPOINT", "").strip().rstrip("/"))
            if judge and judge.get("provider") == "profllm" else None)


def save_context(prompt, objective, converter, rubric):
    from .scoring import build_scoring_context
    context = build_scoring_context(objective, prompt.response, prompt.prompt_text,
                                    prompt.converter_used, rubric, build_response_decoder(converter),
                                    judge=resolve_role("JUDGE"))
    prompt.scoring_context_json = json.dumps(context, ensure_ascii=False)
    return context


def retry_blocker(prompt, *, include_running=True):
    if (by_slug(prompt.attack_slug) or {}).get("kind") not in ("static", "dynamic"):
        # Static rows save their judge request at capture; multi-turn turns save
        # it when the row is written (scoring.assess_turn). Azure scans do not.
        return "Scoring recovery supports saved static responses and multi-turn turns only."
    if not prompt.counts_toward_stats:
        return "No usable saved target response."
    # Valid low scores and valid refusals are outcomes, never recovery targets.
    if prompt.severity is not None and 1 <= prompt.severity <= 5:
        return "This response already has a score."
    context = prompt.scoring_context
    required = {"objective", "request", "request_sha256", "response_sha256", "prompt_sha256",
                "converter", "parser_version", "success_threshold", "judge", "rubric"}
    if not context or not required.issubset(context) or context.get("version") != 1:
        return "Original scoring inputs were not saved; same-input recovery is unavailable."
    if context["parser_version"] != JUDGE_PARSER_VERSION:
        return "The saved scoring parser differs from this version."
    try:
        if (text_hash(context["request"]) != context["request_sha256"]
                or text_hash(prompt.response) != context["response_sha256"]
                or text_hash(prompt.prompt_text) != context["prompt_sha256"]
                or prompt.converter_used != context["converter"]):
            return "Saved scoring evidence no longer matches this response."
    except (AttributeError, TypeError):
        return "Saved scoring evidence is invalid."
    judge = context["judge"]
    if not isinstance(judge, dict) or not all(judge.get(k) for k in ("provider", "model", "key_env")):
        return "The original judge configuration was not recorded."
    if not isinstance(context["success_threshold"], int) or not 1 <= context["success_threshold"] <= 5:
        return "The original success threshold is invalid."
    if include_running and any(e.status == "running" for e in prompt.current_scoring_evaluations):
        return "Scoring is already in progress."
    return None


def new_evaluation(prompt, *, previous_status=None):
    evaluation = ScoringEvaluation(prompt_id=prompt.id, context_json=prompt.scoring_context_json,
                                   previous_benchmark_status=previous_status, status="running")
    db.session.add(evaluation)
    db.session.commit()  # capture, request and pending evaluation survive a crash
    return evaluation


async def evaluate(prompt, evaluation, judge, *, initial_call=None):
    context = json.loads(evaluation.context_json)
    start = time.perf_counter()
    try:
        with request_history(prompt.benchmark_id, "judge", prompt=prompt, evaluation=evaluation):
            if initial_call is not None:
                verdict, severity, succeeded = await initial_call()
            else:
                verdict, severity, succeeded = await judge_request(
                    judge, context["request"], success_threshold=context["success_threshold"])
    except RequestAccountingError:
        raise
    except Exception as exc:
        # Do not persist exception strings that may contain URLs or credentials.
        verdict, severity, succeeded = f"judge-error: {type(exc).__name__}", None, False
    evaluation.elapsed_ms = max(0, int((time.perf_counter() - start) * 1000))
    evaluation.finished_at = _utcnow()
    evaluation.verdict, evaluation.severity = verdict, severity
    evaluation.severity_source = severity_provenance(verdict, severity)
    evaluation.succeeded = succeeded
    evaluation.status = ("scored" if severity is not None else
                         "error" if verdict.startswith("judge-error:") else "invalid")
    # Evidence is unchanged; only assessment fields get the latest outcome.
    prompt.verdict, prompt.severity, prompt.succeeded = verdict, severity, succeeded
    prompt.severity_source = evaluation.severity_source
    db.session.commit()


def claim_retry_status(benchmark, message):
    # The caller's ORM object may predate a worker's final commit. Read a
    # scalar from the database and claim exactly the status we will restore.
    previous = db.session.query(Benchmark.status).filter_by(id=benchmark.id).scalar()
    if previous not in TERMINAL:
        db.session.rollback()
        raise ValueError(message)
    claimed = Benchmark.query.filter(Benchmark.id == benchmark.id,
                                     Benchmark.status == previous).update(
        {Benchmark.status: "scoring"}, synchronize_session=False)
    if not claimed:
        db.session.rollback()
        raise ValueError(message)
    return previous


def claim_retry(benchmark, prompt):
    reason = retry_blocker(prompt)
    if reason:
        raise ValueError(reason)
    previous = claim_retry_status(
        benchmark, "Wait until this benchmark stops before retrying scoring.")
    # Evaluation creation and benchmark claim are committed together.
    return new_evaluation(prompt, previous_status=previous)


def fail_evaluation(evaluation, message):
    evaluation.status = "error"
    evaluation.verdict = message
    evaluation.finished_at = _utcnow()
    db.session.commit()


async def retry_evaluation(evaluation_id):
    evaluation = db.session.get(ScoringEvaluation, evaluation_id)
    if not evaluation or evaluation.status != "running":
        return
    prompt = db.session.get(Prompt, evaluation.prompt_id)
    if not prompt:
        return
    benchmark = prompt.benchmark
    token = None
    active_token = active_benchmark.set(benchmark.id)
    judge = None
    try:
        reason = retry_blocker(prompt, include_running=False)
        if reason or prompt.scoring_context_json != evaluation.context_json:
            raise RecoveryUnavailable(reason or "Saved scoring inputs changed after the retry was queued.")
        context = json.loads(evaluation.context_json)
        saved = context["judge"]
        if endpoint_identity(saved) != context.get("endpoint_sha256"):
            raise RecoveryUnavailable("The saved judge endpoint differs from the configured endpoint.")
        config = {"provider": saved["provider"], "model": saved["model"],
                  "api_key_env": saved["key_env"]}
        if saved.get("inline_key"):
            key = (benchmark.llm_config.get("JUDGE") or {}).get("api_key")
            if not key:
                raise RecoveryUnavailable("The original judge credential is unavailable.")
            config["api_key"] = key
        token = llm_overrides.set({"JUDGE": config})
        judge = build_judge_llm()
        if judge is None:
            raise RecoveryUnavailable("The recorded judge is unavailable.")
        set_activity(benchmark.id, "judging", "retrying scoring on the saved response")
        await evaluate(prompt, evaluation, judge)
    except RequestAccountingError:
        evaluation.status = "interrupted"
        evaluation.verdict = "Local request accounting failed; no scoring result persisted. Inspect the request ledger before retrying."
        evaluation.finished_at = _utcnow()
        db.session.commit()
    except Exception as exc:
        # User-facing preflight reasons are fixed strings; external exceptions
        # are reduced to type to avoid leaking provider details.
        message = str(exc) if isinstance(exc, RecoveryUnavailable) else type(exc).__name__
        fail_evaluation(evaluation, f"Scoring retry unavailable: {message}")
    finally:
        if token is not None:
            llm_overrides.reset(token)
        active_benchmark.reset(active_token)
        if judge is not None and getattr(judge, "aclose", None):
            try:
                await judge.aclose()
            except Exception:
                pass
        totals = benchmark.compute_totals()
        benchmark.total_prompts, benchmark.total_successes = totals["counted"], totals["successes"]
        benchmark.avg_latency_ms = totals["avg_latency_ms"]
        benchmark.status = evaluation.previous_benchmark_status or "interrupted"
        clear_activity(benchmark.id)
        db.session.commit()
    return True  # the wrapper must not release a later worker's claim


def start_scoring_retry(app, evaluation_id):
    evaluation = db.session.get(ScoringEvaluation, evaluation_id)
    prompt = db.session.get(Prompt, evaluation.prompt_id) if evaluation else None
    if not prompt:
        raise RecoveryUnavailable("The queued scoring evaluation is unavailable.")
    benchmark_id, previous = prompt.benchmark_id, evaluation.previous_benchmark_status

    def work():
        with app.app_context():
            released = False
            try:
                released = asyncio.run(retry_evaluation(evaluation_id)) is True
            finally:
                # Even a disappeared/cancelled evaluation or an unexpected
                # worker exception must release the benchmark claim. Do not
                # restart target execution to recover a scoring-only job.
                try:
                    if not released:
                        db.session.rollback()
                        pending = db.session.get(ScoringEvaluation, evaluation_id)
                        if pending and pending.status == "running":
                            pending.status = "interrupted"
                            pending.finished_at = _utcnow()
                            pending.verdict = "Scoring worker ended without a result; delivery/outcome unknown."
                        benchmark = db.session.get(Benchmark, benchmark_id)
                        if benchmark and benchmark.status == "scoring":
                            benchmark.status = previous or "interrupted"
                        clear_activity(benchmark_id)
                        db.session.commit()
                finally:
                    db.session.remove()
    threading.Thread(target=work, name=f"score-{evaluation_id}", daemon=True).start()
